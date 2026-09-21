"""
Run-2 analysis (Addendum A of notes/falsification.md).  For one model and one transfer direction it computes,
from schema-v2 prediction files:

  H1/H8  coverage gap on the target (raw score) at alpha in {.05,.10,.20}, bootstrap CI
  H4     model-uncertainty-normalised score: gap, in-domain check, region-area ratio
  H2/H3  re-run with LABEL-FREE factors only (Addendum A0) + oracle (GT-derived) factors as a diagnostic
  H5     label budget (direct / pooled / shrinkage) ;   H5b  coverage audit power + false alarm
  diag   quantile-ratio profile; conditional coverage by predicted-scale bin (raw vs normalised)
  H6/H7  (forward direction, AV2 models) per-city coverage + label-free shift monitor across city pairs

Usage
  python src/analyze_run2.py --models av2_cpu_v1 av2_valsplit_v1            # forward: av2cal -> {av2test, ns}
  python src/analyze_run2.py --models ns_gpu_v1 --direction reverse         # nscal -> {nstest, av2}

Outputs  results/run2/<model>__<direction>.json  (+ cities__<model>.json for forward)
Every number in the paper is produced by paper/make_results.py from these files.
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
from conformal import (SplitConformal, WeightedSplitConformal, NormalizedSplitConformal,  # noqa: E402
                       GroupConditionalConformal, domain_classifier_weights, split_conformal_quantile)
from attribution import fit_shift_regression  # noqa: E402
from shift_factors import FACTOR_NAMES, FACTOR_NAMES_LF  # noqa: E402
from solutions import (normalized_scores, region_area_proxy, label_budget, audit_power,  # noqa: E402
                       output_features, domain_auc)

ALPHAS = (0.05, 0.10, 0.20)
OUT = ROOT / "results" / "run2"
PREDS = ROOT / "results" / "preds2"


# ------------------------------------------------------------------------------------------ io
def load(model: str, name: str) -> dict | None:
    p = PREDS / model / f"{name}.npz"
    if not p.exists():
        return None
    d = dict(np.load(p, allow_pickle=True))
    if "pred_scale" not in d:
        raise RuntimeError(f"{p} is not schema v2")
    d["scores_norm"] = normalized_scores(d["pred_trajs"], d["gt"], d["gt_mask"], d["pred_scale"])
    return d


def concat(ds: list[dict]) -> dict:
    keys = [k for k in ds[0] if isinstance(ds[0][k], np.ndarray) and ds[0][k].ndim >= 1 and k not in
            ("factor_names", "factor_names_lf", "schema")]
    out = {k: np.concatenate([d[k] for d in ds]) for k in keys}
    out["factor_names"] = ds[0]["factor_names"]; out["factor_names_lf"] = ds[0]["factor_names_lf"]
    return out


def minade_topk(d: dict, k=5, step=5) -> float:
    """nuScenes-challenge-style minADE_k: top-k modes by probability, 2 Hz samples of the 10 Hz output."""
    P, pr, G, M = d["pred_trajs"], d["pred_probs"], d["gt"], d["gt_mask"]
    idx = np.arange(step - 1, P.shape[2], step)
    top = np.argsort(-pr, axis=1)[:, :k]
    vals = []
    for i in range(len(P)):
        m = M[i, idx]
        if m.sum() == 0:
            continue
        e = np.linalg.norm(P[i, top[i]][:, idx] - G[i, idx][None], axis=-1)
        vals.append((e * m).sum(1).min() / m.sum())
    return float(np.mean(vals))


# ------------------------------------------------------------------------------------------ helpers
def boot_gap(cal_s, ev_s, alpha, B=500, seed=0):
    """Point gap (nominal - coverage) with calibration set = all of cal_s, plus bootstrap 95% CI that
    resamples BOTH calibration and evaluation sets."""
    q = split_conformal_quantile(cal_s, alpha)
    gap = (1 - alpha) - float((ev_s <= q).mean())
    rng = np.random.default_rng(seed)
    g = np.empty(B)
    for b in range(B):
        qb = split_conformal_quantile(cal_s[rng.integers(0, len(cal_s), len(cal_s))], alpha)
        g[b] = (1 - alpha) - float((ev_s[rng.integers(0, len(ev_s), len(ev_s))] <= qb).mean())
    return {"gap": gap, "coverage": (1 - alpha) - gap, "q": float(q),
            "ci95": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]}


def h1_h4(cal, same, tgt):
    """Raw vs normalised score: gap on same-domain test and on target, and region-area ratios."""
    out = {}
    K = cal["pred_probs"].shape[1]
    for a in ALPHAS:
        r = {}
        for tag, key in (("raw", "scores"), ("norm", "scores_norm")):
            r[tag] = {"same": boot_gap(cal[key], same[key], a, B=300), "target": boot_gap(cal[key], tgt[key], a, B=300)}
        q_raw, q_n = r["raw"]["target"]["q"], r["norm"]["target"]["q"]
        # area proxy on the TARGET set: normalised region uses target's predicted scales, raw uses constant radius
        A_raw = region_area_proxy(q_raw, len(tgt["scores"]), K).mean()
        A_norm = region_area_proxy(q_n, len(tgt["scores"]), K, tgt["pred_scale"]).mean()
        r["area_ratio_norm_over_raw_target"] = float(A_norm / A_raw)
        As = region_area_proxy(r["raw"]["same"]["q"], len(same["scores"]), K).mean()
        An = region_area_proxy(r["norm"]["same"]["q"], len(same["scores"]), K, same["pred_scale"]).mean()
        r["area_ratio_norm_over_raw_same"] = float(An / As)
        # FAIR efficiency: compare the two score functions at EXACT nominal coverage on the target (oracle
        # threshold from target labels).  A region that under-covers is not comparable with one that does not.
        qo_raw = float(np.quantile(tgt["scores"], 1 - a)); qo_n = float(np.quantile(tgt["scores_norm"], 1 - a))
        Ao_raw = region_area_proxy(qo_raw, len(tgt["scores"]), K).mean()
        Ao_n = region_area_proxy(qo_n, len(tgt["scores"]), K, tgt["pred_scale"]).mean()
        r["area_ratio_norm_over_raw_at_exact_target_coverage"] = float(Ao_n / Ao_raw)
        out[f"{a:.2f}"] = r
    return out


def h2_h3_label_free(cal, same, tgt, alpha=0.10, seeds=10):
    """H2 regression + H3 recalibration using ONLY label-free factors (and, for H2, also the oracle set)."""
    S = np.r_[same["scores"], tgt["scores"]]
    res = {"H2": {}}
    res["H2"]["label_free_pooled_adj_r2"] = fit_shift_regression(S, np.vstack([same["feats_lf"], tgt["feats_lf"]]))["adj_r2"]
    res["H2"]["oracle_gt_derived_pooled_adj_r2"] = fit_shift_regression(S, np.vstack([same["feats"], tgt["feats"]]))["adj_r2"]
    res["H2"]["label_free_within_source_adj_r2"] = fit_shift_regression(same["scores"], same["feats_lf"][:, 1:])["adj_r2"]
    res["H2"]["label_free_within_target_adj_r2"] = fit_shift_regression(tgt["scores"], tgt["feats_lf"][:, 1:])["adj_r2"]
    # + model-output features (exploratory)
    of_s = output_features(same["pred_probs"], same["pred_scale"], same["pred_trajs"])
    of_t = output_features(tgt["pred_probs"], tgt["pred_scale"], tgt["pred_trajs"])
    res["H2"]["label_free_plus_output_pooled_adj_r2_EXPLORATORY"] = fit_shift_regression(
        S, np.hstack([np.vstack([same["feats_lf"], tgt["feats_lf"]]), np.vstack([of_s, of_t])]))["adj_r2"]

    cal_of = output_features(cal["pred_probs"], cal["pred_scale"], cal["pred_trajs"])
    acc = {k: [] for k in ("uncorrected", "weighted", "normalized", "group")}
    rad = {k: [] for k in acc}; removal = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        idx = rng.permutation(len(cal["scores"]))[: len(cal["scores"]) // 2]
        cs, cf = cal["scores"][idx], cal["feats_lf"][idx]
        ts, tf = tgt["scores"], tgt["feats_lf"]
        r = SplitConformal(alpha).calibrate(cs).evaluate(ts); acc["uncorrected"].append(r["coverage"]); rad["uncorrected"].append(r["q_hat"])
        w = domain_classifier_weights(cf, tf, seed=sd)
        r = WeightedSplitConformal(alpha).calibrate(cs, w).evaluate(ts); acc["weighted"].append(r["coverage"]); rad["weighted"].append(r["q_hat"])
        r = NormalizedSplitConformal(alpha).calibrate(cs, cf).evaluate(ts, tf); acc["normalized"].append(r["coverage"]); rad["normalized"].append(r["q_hat_raw_equiv"])
        q1, q2 = np.percentile(cf[:, 1], [33, 66])
        r = GroupConditionalConformal(alpha).calibrate(cs, np.digitize(cf[:, 1], [q1, q2])).evaluate(ts, np.digitize(tf[:, 1], [q1, q2]))
        acc["group"].append(r["coverage"]); rad["group"].append(r["q_hat"])
        gb = (1 - alpha) - acc["uncorrected"][-1]; ga = (1 - alpha) - acc["weighted"][-1]
        removal.append((gb - ga) / gb if abs(gb) > 1e-6 else float("nan"))
    res["H3_label_free"] = {k: {"coverage": float(np.mean(v)), "sd": float(np.std(v)), "gap": (1 - alpha) - float(np.mean(v)),
                                "radius": float(np.mean(rad[k]))} for k, v in acc.items()}
    res["H2"]["label_free_reweighting_removal_fraction"] = float(np.nanmean(removal))
    return res


def h5(cal, tgt, same):
    """Label budget (alpha=.10, 200 draws) + audit power on the target and false alarm on same-domain data."""
    out = {"label_budget": {}, "audit": {}}
    for a in ALPHAS:
        lb = label_budget(cal["scores"], tgt["scores"], alpha=a, draws=200, seed=1)
        out["label_budget"][f"{a:.2f}"] = lb
    q = split_conformal_quantile(cal["scores"], 0.10)
    out["audit"]["target"] = audit_power(q, tgt["scores"], alpha=0.10, draws=500, seed=2)
    out["audit"]["same_domain"] = audit_power(q, same["scores"], alpha=0.10, draws=500, seed=3)
    # exploratory: the same audit/label-budget on the normalised score
    qn = split_conformal_quantile(cal["scores_norm"], 0.10)
    out["audit"]["target_normalised_EXPLORATORY"] = audit_power(qn, tgt["scores_norm"], alpha=0.10, draws=500, seed=4)
    return out


def diagnostics(cal, same, tgt, alpha=0.10):
    q = split_conformal_quantile(cal["scores"], alpha); qn = split_conformal_quantile(cal["scores_norm"], alpha)
    ps = np.arange(50, 100, 5).tolist() + [97, 99]
    ratio = {str(p): float(np.percentile(tgt["scores"], p) / np.percentile(same["scores"], p)) for p in ps}
    # conditional coverage by bins of the (label-free) top-mode predicted scale, bins = source quintiles
    u = lambda d: d["pred_scale"][np.arange(len(d["scores"])), d["pred_probs"].argmax(1)]
    edges = np.percentile(u(cal), [20, 40, 60, 80])
    rows = []
    for b in range(5):
        r = {"bin": b}
        for nm, d in (("same", same), ("target", tgt)):
            m = np.digitize(u(d), edges) == b
            r[f"{nm}_n"] = int(m.sum())
            r[f"{nm}_cov_raw"] = float((d["scores"][m] <= q).mean()) if m.sum() else None
            r[f"{nm}_cov_norm"] = float((d["scores_norm"][m] <= qn).mean()) if m.sum() else None
        rows.append(r)
    return {"quantile_ratio_target_over_source": ratio, "coverage_by_predicted_scale_quintile": rows,
            "spearman_predscale_vs_score": {
                "source": float(_spearman(u(same), same["scores"])), "target": float(_spearman(u(tgt), tgt["scores"]))}}


def _spearman(a, b):
    from scipy.stats import spearmanr
    return spearmanr(a, b)[0]


# ------------------------------------------------------------------------------------------ cities
def load_city_map(sids) -> dict:
    cache = OUT / "city_map_av2val.json"
    m = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [s for s in set(map(str, sids)) if s not in m]
    if todo:
        import pyarrow.parquet as pq
        t0 = time.time()
        for s in todo:
            f = ROOT / "data" / "argoverse2" / "val" / s / f"scenario_{s}.parquet"
            m[s] = pq.read_table(f, columns=["city"]).column(0)[0].as_py()
        OUT.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(m))
        print(f"[cities] read {len(todo)} files in {time.time()-t0:.0f}s")
    return m


def cities(av2: dict, alpha=0.10, min_n=300, seeds=10) -> dict:
    cm = load_city_map(av2["scenario_id"]); city = np.array([cm[str(s)] for s in av2["scenario_id"]])
    names = [c for c in sorted(set(city)) if (city == c).sum() >= min_n]
    out = {"n_per_city": {c: int((city == c).sum()) for c in sorted(set(city))}}
    # H7: one common calibration (random half of the pooled set), per-city coverage on the other half
    cov = {c: [] for c in names}
    for sd in range(seeds):
        rng = np.random.default_rng(sd); perm = rng.permutation(len(city)); cal_i, ev_i = perm[: len(perm)//2], perm[len(perm)//2:]
        q = split_conformal_quantile(av2["scores"][cal_i], alpha)
        for c in names:
            m = ev_i[city[ev_i] == c]; cov[c].append(float((av2["scores"][m] <= q).mean()))
    out["H7_per_city_coverage"] = {c: {"coverage": float(np.mean(v)), "gap": (1 - alpha) - float(np.mean(v)),
                                       "n_eval_per_seed": int(((city == c).sum()) // 2)} for c, v in cov.items()}
    # H6: ordered city pairs, label-free monitor vs realised gap
    F = np.hstack([av2["feats_lf"], output_features(av2["pred_probs"], av2["pred_scale"], av2["pred_trajs"])])
    rows = []
    for i in names:
        qi = split_conformal_quantile(av2["scores"][city == i], alpha)
        for j in names:
            if i == j:
                continue
            gap = (1 - alpha) - float((av2["scores"][city == j] <= qi).mean())
            rows.append({"src": i, "tgt": j, "gap": gap, "abs_gap": abs(gap),
                         "auc": domain_auc(F[city == i], F[city == j], seed=0),
                         "auc_output_only": domain_auc(F[city == i][:, -3:], F[city == j][:, -3:], seed=0)})
    from scipy.stats import spearmanr
    T = np.array([r["auc"] for r in rows]); G = np.array([r["abs_gap"] for r in rows])
    rho = float(spearmanr(T, G)[0])
    rng = np.random.default_rng(0)
    perm_p = float(np.mean([abs(spearmanr(T, rng.permutation(G))[0]) >= abs(rho) for _ in range(2000)]))
    out["H6"] = {"pairs": rows, "spearman_auc_vs_abs_gap": rho, "perm_p": perm_p, "n_pairs": len(rows),
                 "spearman_output_only": float(spearmanr([r["auc_output_only"] for r in rows], G)[0])}
    return out


# ------------------------------------------------------------------------------------------ main
def run_model(model: str, direction: str) -> None:
    if direction == "forward":
        cal, same = load(model, "av2cal"), load(model, "av2test")
        tgt = load(model, "ns")
        tgt_name = "ns"
    else:
        cal, same = load(model, "nscal"), load(model, "nstest")
        a1, a2 = load(model, "av2cal"), load(model, "av2test")
        tgt = concat([a1, a2]) if a1 is not None and a2 is not None else None
        tgt_name = "av2"
    if cal is None or same is None or tgt is None:
        print(f"[skip] {model}/{direction}: missing prediction sets"); return
    journal.event(f"run2_{model}_{direction}", "attempt_start", f"n cal/same/tgt = {len(cal['scores'])}/{len(same['scores'])}/{len(tgt['scores'])}")
    t0 = time.time()
    res = {"model": model, "direction": direction, "target": tgt_name,
           "n": {"cal": len(cal["scores"]), "same": len(same["scores"]), "target": len(tgt["scores"])},
           "base_model": {"minADE5_2Hz_same": minade_topk(same), "minADE5_2Hz_target": minade_topk(tgt),
                          "same_miss_rate_FDE2m": float(np.mean([np.linalg.norm(same["pred_trajs"][i, :, -1] - same["gt"][i, -1], axis=-1).min() > 2.0 for i in range(len(same["scores"]))]))}}
    res["H1_H4"] = h1_h4(cal, same, tgt)
    res.update(h2_h3_label_free(cal, same, tgt))
    res.update(h5(cal, tgt, same))
    res["diagnostics"] = diagnostics(cal, same, tgt)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{model}__{direction}.json"
    p.write_text(json.dumps(res, indent=1))
    if direction == "forward":
        full_av2 = concat([cal, same])
        (OUT / f"cities__{model}.json").write_text(json.dumps(cities(full_av2), indent=1))
    journal.event(f"run2_{model}_{direction}", "done", f"{p.relative_to(ROOT)} in {time.time()-t0:.0f}s; "
                  f"raw gap@.10={res['H1_H4']['0.10']['raw']['target']['gap']:+.4f} norm gap={res['H1_H4']['0.10']['norm']['target']['gap']:+.4f}")
    h = res["H1_H4"]["0.10"]
    print(f"{model}/{direction}: raw gap {h['raw']['target']['gap']:+.4f} CI{np.round(h['raw']['target']['ci95'],4).tolist()} | "
          f"norm gap {h['norm']['target']['gap']:+.4f} (same {h['norm']['same']['gap']:+.4f}) | area ratio {h['area_ratio_norm_over_raw_target']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    a = ap.parse_args()
    for m in a.models:
        run_model(m, a.direction)
