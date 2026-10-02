"""
Revision R1, steps 1.4 / 1.6 (Addendum D, D-2): scene-level repair with the CORRECTED betting p-value and DISJOINT
labelled / evaluation pools.  For each transfer pair:

  labelled pool  -> draw m whole scenes without replacement  (m in MS, DRAWS draws)
  evaluation pool -> R(q_hat) = scene-averaged miscoverage, computed EXACTLY on the full evaluation pool
  reported: P[R(q_hat) <= alpha] with a Wilson 95% interval, fraction of infinite thresholds, median area vs the
  scene-averaged oracle, median finite radius in metres; methods: betting-LTT, HB-LTT, agent-level C-or-R, pooled
  agent quantile ("direct"), one-agent-per-scene split CP (Dunn et al. 2023, hierarchical baseline).
  Each rule also at the tolerance target alpha' = 0.12 (sensitivity row, Addendum D-2).

Usage:  python src/analyze_rev1.py --pair av2_to_nuscenes        (see PAIRS)
Output: results/rev1/scene_<pair>.json     marker: results/rev1/1.6_<pair>.ok
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402
import solutions  # noqa: E402
from conformal import split_conformal_quantile  # noqa: E402
from solutions import BET_CAP, certify_or_recalibrate, q_direct  # noqa: E402

ALPHA, DELTA = 0.10, 0.10
ALPHA_TOL = 0.12
MS = (20, 30, 40, 60, 80, 100, 150, 200, 300)
DRAWS = 1000
PREDS = ROOT / "results" / "preds2"

# pair -> (model, source-calibration file, labelled-pool files (list), evaluation-pool file)
PAIRS = {
    "av2_to_nuscenes": ("av2_gpu_full", "av2cal", [f"nstrain_c{i:02d}" for i in range(11)], "ns"),
    "av2_to_waymo":    ("av2_gpu_full", "av2cal", ["wm_av2gpu_cal"], "wm_av2gpu_test"),
    "ns_to_waymo":     ("ns_gpu_full", "nscal", ["wm_nsgpu_cal"], "wm_nsgpu_test"),
    "ns_to_av2":       ("ns_gpu_full", "nscal", ["av2cal"], "av2test"),
}


def load_scores(model: str, names: list[str]):
    S, C = [], []
    for n in names:
        d = np.load(PREDS / model / f"{n}.npz", allow_pickle=True)
        S.append(d["scores"].astype(float))
        C.append(np.array([(re.search(r"scene-\d+", str(s)) or [str(s)])[0] if re.search(r"scene-\d+", str(s)) else str(s)
                           for s in d["scenario_id"]]))
    return np.concatenate(S), np.concatenate(C)


class Pool:
    """Scores grouped by scene, plus the exact scene-averaged miscoverage R(q) for that pool."""

    def __init__(self, scores, clusters):
        u, inv = np.unique(clusters, return_inverse=True)
        self.scene_ids = u
        self.groups = [np.sort(scores[inv == c]) for c in range(len(u))]
        self.raw = [scores[inv == c] for c in range(len(u))]
        n = np.array([len(g) for g in self.groups], float)
        order = np.argsort(scores)
        w = (1.0 / (n[inv] * len(u)))[order]
        self._s, self._cw = scores[order], np.cumsum(w)

    def R(self, q: float) -> float:
        if not np.isfinite(q):
            return 0.0
        i = np.searchsorted(self._s, q, side="right")
        return float(1.0 - (self._cw[i - 1] if i > 0 else 0.0))

    def oracle(self, alpha: float) -> float:
        i = int(np.searchsorted(self._cw, 1.0 - alpha - 1e-12))
        return float(self._s[min(i, len(self._s) - 1)])


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def fast_scene_ltt(q_src: float, groups: list, alpha: float, delta: float, pval: str):
    """Vectorised equivalent of solutions.scene_ltt_threshold (checked against it in --selfcheck)."""
    m = len(groups)
    grid = q_src * np.exp(np.linspace(np.log(4.0), np.log(0.5), 300))
    Q = np.concatenate([[q_src], grid])
    L = np.stack([1.0 - np.searchsorted(g, Q, side="right") / len(g) for g in groups])        # (m, G+1)
    if pval == "hb":
        r = L.mean(0)
        a = np.minimum(r, alpha)
        with np.errstate(divide="ignore", invalid="ignore"):
            h1 = np.where(a <= 0, -np.log(1 - alpha),
                          a * np.log(np.where(a > 0, a, 1) / alpha) + (1 - a) * np.log((1 - a) / (1 - alpha)))
        p = np.minimum(np.exp(-m * h1), np.e * binom.cdf(np.ceil(m * r), m, alpha))
    else:
        mu = np.full(Q.shape, alpha / 2); var = np.full(Q.shape, alpha / 4); n = 1.0
        k = np.ones(Q.shape); kmax = np.ones(Q.shape)
        for x in L:                                              # scenes in draw order (predictable bets)
            edge = alpha - mu
            lam = np.where(edge <= 0, 0.0, np.minimum(edge / (var + edge * edge), BET_CAP / (1.0 - alpha)))
            k = k * (1.0 + lam * (alpha - x))
            assert (k >= 0).all(), "betting capital went negative"
            kmax = np.maximum(kmax, k)
            n += 1; d = x - mu; mu = mu + d / n; var = var + (d * (x - mu) - var) / n
        p = np.minimum(1.0, 1.0 / kmax)
    if p[0] <= delta / 2:
        return float(q_src)
    rej = p[1:] <= delta / 2
    if not rej[0]:
        return float("inf")
    first_fail = int(np.argmin(rej)) if not rej.all() else len(rej)
    return float(grid[first_fail - 1])


def selfcheck(rng):
    groups_all = [np.sort(np.exp(rng.normal(1.0 + rng.normal(0, .35), .45, int(rng.integers(1, 70))))) for _ in range(60)]
    q_src = float(np.quantile(np.concatenate(groups_all), 0.8))
    bad = 0
    for _ in range(25):
        idx = rng.permutation(60)[:int(rng.choice([20, 40, 60]))]
        lab = [groups_all[i] for i in idx]
        for pv in ("hb", "wsr"):
            a = fast_scene_ltt(q_src, lab, ALPHA, DELTA, pv)
            b, _ = solutions.scene_ltt_threshold(q_src, lab, ALPHA, DELTA, pval=pv)
            bad += not (a == b or (np.isinf(a) and np.isinf(b)))
    return bad == 0


def run_pair(pair: str, draws: int = DRAWS, ms=MS, seed: int = 0):
    model, cal_name, lab_names, ev_name = PAIRS[pair]
    t0 = time.time()
    cal_s, _ = load_scores(model, [cal_name])
    lab_s, lab_c = load_scores(model, lab_names)
    ev_s, ev_c = load_scores(model, [ev_name])
    lab, ev = Pool(lab_s, lab_c), Pool(ev_s, ev_c)
    assert not (set(lab.scene_ids) & set(ev.scene_ids)), "labelled and evaluation pools share a scene"
    q_src = float(split_conformal_quantile(cal_s, ALPHA))
    rng = np.random.default_rng(seed)
    out = {"pair": pair, "model": model, "alpha": ALPHA, "alpha_tol": ALPHA_TOL, "delta": DELTA, "draws": draws,
           "n_label_scenes": len(lab.scene_ids), "n_eval_scenes": len(ev.scene_ids), "bet_cap": BET_CAP,
           "q_src": q_src, "R_src_on_eval": ev.R(q_src),
           "q_oracle": {"0.10": ev.oracle(ALPHA), "0.12": ev.oracle(ALPHA_TOL)}, "by_m": {}}
    methods = ("bet", "hb", "bet_tol", "hb_tol", "agent_cor", "direct", "hier_sub1")
    for m in [m for m in ms if m <= len(lab.scene_ids)]:
        acc = {k: {"ok": 0, "inf": 0, "q": [], "area": []} for k in methods}
        for _ in range(draws):
            pick = rng.permutation(len(lab.scene_ids))[:m]
            g = [lab.groups[i] for i in pick]
            raw = [lab.raw[i] for i in pick]
            pooled = np.concatenate(raw)
            qs = {
                "bet": fast_scene_ltt(q_src, g, ALPHA, DELTA, "wsr"),
                "hb": fast_scene_ltt(q_src, g, ALPHA, DELTA, "hb"),
                "bet_tol": fast_scene_ltt(q_src, g, ALPHA_TOL, DELTA, "wsr"),
                "hb_tol": fast_scene_ltt(q_src, g, ALPHA_TOL, DELTA, "hb"),
                "agent_cor": certify_or_recalibrate(q_src, pooled, ALPHA, DELTA)[0],
                "direct": float(q_direct(pooled, ALPHA)),
                "hier_sub1": float(split_conformal_quantile(np.array([r[rng.integers(len(r))] for r in raw]), ALPHA)),
            }
            for k, q in qs.items():
                tgt_alpha = ALPHA_TOL if k.endswith("_tol") else ALPHA
                q = float(q) if np.isfinite(q) else float("inf")
                acc[k]["ok"] += ev.R(q) <= tgt_alpha
                if np.isinf(q):
                    acc[k]["inf"] += 1
                else:
                    acc[k]["q"].append(q)
                    acc[k]["area"].append((q / ev.oracle(tgt_alpha)) ** 2)
        rowm = {}
        for k, a in acc.items():
            lo, hi = wilson(int(a["ok"]), draws)
            rowm[k] = {"p_valid": a["ok"] / draws, "wilson95": [lo, hi], "frac_infinite": a["inf"] / draws,
                       "median_area_vs_oracle": float(np.median(a["area"])) if a["area"] else None,
                       "median_q_finite_m": float(np.median(a["q"])) if a["q"] else None}
        out["by_m"][str(m)] = rowm
        b = rowm["bet"]
        print(f"  {pair} m={m:3d}: bet valid={b['p_valid']:.3f} [{b['wilson95'][0]:.3f},{b['wilson95'][1]:.3f}] "
              f"inf={b['frac_infinite']:.2f} area={b['median_area_vs_oracle']} | hb inf={rowm['hb']['frac_infinite']:.2f} | "
              f"agent_cor valid={rowm['agent_cor']['p_valid']:.2f} | hier_sub1 valid={rowm['hier_sub1']['p_valid']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=list(PAIRS))
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        ok = selfcheck(np.random.default_rng(3))
        print("selfcheck fast_scene_ltt == scene_ltt_threshold:", "OK" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    journal.event(f"rev1_scene_{a.pair}", "attempt_start", f"draws={a.draws}")
    res = run_pair(a.pair, a.draws)
    (ROOT / "results" / "rev1").mkdir(parents=True, exist_ok=True)
    p = ROOT / "results" / "rev1" / f"scene_{a.pair}.json"
    p.write_text(json.dumps(res, indent=1))
    (ROOT / "results" / "rev1" / f"1.6_{a.pair}.ok").write_text("ok")
    journal.event(f"rev1_scene_{a.pair}", "done", str(p.relative_to(ROOT)))
