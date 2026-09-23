"""
Addendum-B analyses that need only existing predictions (notes/falsification.md, Addendum B):

  B5 / H14  certify-or-recalibrate (C-or-R) vs direct SCP vs PAC-only, on the shifted target and on the
            same-domain control; plus an EXPLORATORY variant (source threshold calibrated with slack, alpha/2),
            added after the synthetic check showed certification needs slack -- labelled exploratory in output.
  B6 / H15  adaptive conformal inference (ACI) on random-order target label streams
  B7        conditional coverage by UniTraj trajectory type, speed tercile, Kalman-difficulty tercile
  B9        KS distance between source-cal and target score laws (raw and normalised) -- the H4 validity condition

Usage
  python src/analyze_addB.py --models av2_cpu_v1 av2_valsplit_v1 av2_cpu_v2
  python src/analyze_addB.py --models ns_cpu_v1 --direction reverse
Output  results/addB/<model>__<direction>.json   (consumed by paper/make_results.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402
from analyze_run2 import load, concat  # noqa: E402
from conformal import split_conformal_quantile  # noqa: E402
from solutions import q_direct, q_pac, certify_or_recalibrate, aci_stream  # noqa: E402

OUT = ROOT / "results" / "addB"
ANNOT = ROOT / "results" / "preds2" / "_annot"
ALPHA, DELTA = 0.10, 0.10
KS = (100, 250, 500, 1000, 2500)
TRAJ_TYPES = ["stationary", "straight", "straight_right", "straight_left",
              "right_u_turn", "right_turn", "left_u_turn", "left_turn"]


# ------------------------------------------------------------------------------------------ io
def attach_annot(d: dict, set_names: list[str]) -> dict:
    """Join model-independent scene annotations by POSITION (same dataset, same order, shuffle=False),
    asserting scenario ids match row by row so a silent misalignment is impossible."""
    parts = [dict(np.load(ANNOT / f"{s}.npz", allow_pickle=True)) for s in set_names]
    sid = np.concatenate([p["scenario_id"] for p in parts])
    if len(sid) != len(d["scores"]) or not np.array_equal(sid, d["scenario_id"]):
        raise RuntimeError(f"annotation/prediction misalignment for {set_names}: {len(sid)} vs {len(d['scores'])}")
    for k in ("trajectory_type", "kalman_difficulty", "speed_now"):
        d[k] = np.concatenate([p[k] for p in parts])
    return d


def sets_for(model: str, direction: str):
    if direction == "forward":
        cal = attach_annot(load(model, "av2cal"), ["av2cal"])
        same = attach_annot(load(model, "av2test"), ["av2test"])
        tgt = attach_annot(load(model, "ns"), ["ns"])
        return cal, same, tgt, "ns"
    cal = attach_annot(load(model, "nscal"), ["nscal"])
    same = attach_annot(load(model, "nstest"), ["nstest"])
    tgt = attach_annot(concat([load(model, "av2cal"), load(model, "av2test")]), ["av2cal", "av2test"])
    return cal, same, tgt, "av2"


# ------------------------------------------------------------------------------------------ B5
def b5_cor(cal_s, tgt_s, draws=200, seed=0) -> dict:
    """Coverage-guarantee rate of each recalibration rule with k labelled target scenes, evaluated on the
    remaining (disjoint) target scenes.  Area is reported relative to the oracle region (target-calibrated at
    exactly 1-alpha on ALL target scores); raw-score area scales with q^2."""
    rng = np.random.default_rng(seed)
    q_src = split_conformal_quantile(cal_s, ALPHA)
    q_src_slack = split_conformal_quantile(cal_s, ALPHA / 2)           # exploratory
    q_or = float(np.quantile(tgt_s, 1 - ALPHA))
    n = len(tgt_s)
    out = {"alpha": ALPHA, "delta": DELTA, "draws": draws, "q_src": float(q_src), "q_oracle": q_or,
           "source_only_coverage": float((tgt_s <= q_src).mean()), "by_k": {}}
    for k in [k for k in KS if k <= n - 1000]:
        rows = {m: [] for m in ("direct", "pac", "cor", "cor_slack_EXPLORATORY")}
        branch = {"cor": 0, "cor_slack_EXPLORATORY": 0}
        for _ in range(draws):
            perm = rng.permutation(n)
            lab, ev = tgt_s[perm[:k]], tgt_s[perm[k:]]
            q_c, b_c = certify_or_recalibrate(q_src, lab, ALPHA, DELTA)
            q_cs, b_cs = certify_or_recalibrate(q_src_slack, lab, ALPHA, DELTA)
            branch["cor"] += b_c == "certified"; branch["cor_slack_EXPLORATORY"] += b_cs == "certified"
            for m, q in (("direct", q_direct(lab, ALPHA)), ("pac", q_pac(lab, ALPHA, DELTA)),
                         ("cor", q_c), ("cor_slack_EXPLORATORY", q_cs)):
                rows[m].append((float((ev <= q).mean()), (q / q_or) ** 2 if np.isfinite(q) else np.inf))
        res = {}
        for m, r in rows.items():
            cov = np.array([x[0] for x in r]); area = np.array([x[1] for x in r])
            res[m] = {"rate_cov_ge_nominal": float((cov >= 1 - ALPHA).mean()), "mean_cov": float(cov.mean()),
                      "p5_cov": float(np.percentile(cov, 5)),
                      "median_area_vs_oracle": float(np.median(area)), "frac_infinite": float(np.isinf(area).mean())}
        res["cor"]["frac_certified"] = branch["cor"] / draws
        res["cor_slack_EXPLORATORY"]["frac_certified"] = branch["cor_slack_EXPLORATORY"] / draws
        out["by_k"][str(k)] = res
    return out


# ------------------------------------------------------------------------------------------ B6
def b6_aci(cal_s, tgt_s, gammas=(0.005, 0.01, 0.05), streams=50, seed=0, tol=0.02) -> dict:
    rng = np.random.default_rng(seed)
    n = len(tgt_s)
    out = {"alpha": ALPHA, "streams": streams, "tol": tol, "by_gamma": {}}
    orders = [rng.permutation(n) for _ in range(streams)]
    for g in gammas:
        frozen = {str(k): [] for k in KS if k <= n - 1000}
        t_stay, reached = [], 0
        for o in orders:
            s = tgt_s[o]
            qs, errs = aci_stream(cal_s, s, ALPHA, g)
            for k in frozen:                                   # ACI used as a k-label recalibration rule
                kk = int(k)
                frozen[k].append(float((s[kk:] <= qs[kk]).mean()))
            run_cov = 1 - np.cumsum(errs) / np.arange(1, n + 1)    # running (long-run) coverage on the stream
            bad = np.nonzero(np.abs(run_cov - (1 - ALPHA)) > tol)[0]
            t = int(bad[-1] + 1) if len(bad) else 0            # first t after which it STAYS within tol
            if t < n:
                reached += 1; t_stay.append(t)
        out["by_gamma"][str(g)] = {
            "frozen_after_k": {k: {"mean_cov": float(np.mean(v)), "rate_cov_ge_nominal": float(np.mean(np.array(v) >= 1 - ALPHA)),
                                   "p_within_tol": float(np.mean(np.abs(np.array(v) - (1 - ALPHA)) <= tol))}
                               for k, v in frozen.items()},
            "running_cov_labels_to_stay_within_tol_median": float(np.median(t_stay)) if t_stay else None,
            "frac_streams_reaching_it": reached / streams,
        }
    return out


# ------------------------------------------------------------------------------------------ B7
def _wilson(x, n, z=1.96):
    if n == 0:
        return [None, None]
    p = x / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def _cells(cov_mask, groups: dict) -> dict:
    out = {}
    for name, idx in groups.items():
        n = int(idx.sum()); x = int(cov_mask[idx].sum())
        out[name] = {"n": n, "coverage": (x / n) if n else None, "ci95": _wilson(x, n)}
    return out


def b7_conditional(cal, same, tgt) -> dict:
    res = {"alpha": ALPHA}
    for key, tag in (("scores", "raw"), ("scores_norm", "norm")):
        q = split_conformal_quantile(cal[key], ALPHA)
        # tercile cut points from the SOURCE calibration set, applied unchanged to same-domain and target
        sp_cut = np.quantile(cal["speed_now"], [1 / 3, 2 / 3])
        kd_cal = cal["kalman_difficulty"][:, 2]; kd_cut = np.quantile(kd_cal[kd_cal >= 0], [1 / 3, 2 / 3])
        r = {}
        for dom, d in (("same", same), ("target", tgt)):
            cov = d[key] <= q
            tt = d["trajectory_type"]; sp = d["speed_now"]; kd = d["kalman_difficulty"][:, 2]
            r[dom] = {
                "overall": _cells(cov, {"all": np.ones(len(cov), bool)})["all"],
                "trajectory_type": _cells(cov, {nm: tt == i for i, nm in enumerate(TRAJ_TYPES)}),
                "speed_tercile": _cells(cov, {"low": sp <= sp_cut[0], "mid": (sp > sp_cut[0]) & (sp <= sp_cut[1]),
                                              "high": sp > sp_cut[1]}),
                "kalman6s_tercile": _cells(cov, {"easy": (kd >= 0) & (kd <= kd_cut[0]),
                                                 "medium": (kd > kd_cut[0]) & (kd <= kd_cut[1]),
                                                 "hard": kd > kd_cut[1], "undefined": kd < 0}),
            }
        r["cuts"] = {"speed": sp_cut.tolist(), "kalman6s": kd_cut.tolist()}
        res[tag] = r
    return res


# ------------------------------------------------------------------------------------------ B9
def b9_ks(cal, tgt) -> dict:
    from scipy.stats import ks_2samp
    return {tag: {"ks": float(ks_2samp(cal[k], tgt[k]).statistic)}
            for tag, k in (("raw", "scores"), ("norm", "scores_norm"))}


# ------------------------------------------------------------------------------------------ driver
def run(model: str, direction: str) -> None:
    t0 = time.time()
    cal, same, tgt, tname = sets_for(model, direction)
    journal.event(f"addB_{model}_{direction}", "attempt_start", f"n cal/same/tgt={len(cal['scores'])}/{len(same['scores'])}/{len(tgt['scores'])}")
    res = {"model": model, "direction": direction, "target": tname,
           "B5_target": b5_cor(cal["scores"], tgt["scores"]),
           "B5_same_domain_control": b5_cor(cal["scores"], same["scores"], seed=1),
           "B6_target": b6_aci(cal["scores"], tgt["scores"]),
           "B7": b7_conditional(cal, same, tgt),
           "B9": b9_ks(cal, tgt)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{model}__{direction}.json"
    p.write_text(json.dumps(res, indent=1))
    b5 = res["B5_target"]["by_k"].get("1000", {})
    journal.event(f"addB_{model}_{direction}", "done", f"{p.relative_to(ROOT)} in {time.time()-t0:.0f}s; k=1000 rate(cov>=.9): "
                  + ", ".join(f"{m}={v['rate_cov_ge_nominal']:.2f}" for m, v in b5.items()))
    print(f"{model}/{direction}: done in {time.time()-t0:.0f}s -> {p}")
    for k, v in res["B5_target"]["by_k"].items():
        print(f"  k={k:>5}: " + "  ".join(f"{m}: rate={x['rate_cov_ge_nominal']:.2f} area={x['median_area_vs_oracle']:.2f}"
                                          for m, x in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    a = ap.parse_args()
    for m in a.models:
        run(m, a.direction)
