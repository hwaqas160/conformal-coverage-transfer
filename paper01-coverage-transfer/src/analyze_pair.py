"""
Reproducible H1/H2/H3 analysis for ONE (source model -> target dataset) pair, correctly
separating same-domain held-out splits from the true cross-domain target, and writing
everything to results/pair_<source>__<target>.json.

Usage:
  python src/analyze_pair.py --src av2_cpu_v1 --cal av2cal --same av2test --target ns
Expects results/preds/<name>_from_<src>.npz for name in {cal, same, target}.

Also reports a minADE5 on the target using the nuScenes prediction-challenge protocol
(top-5 modes by probability, 2 Hz samples of the 10 Hz output over 6 s) so the base model
can be compared with published nuScenes AutoBot numbers (UniTraj paper: 1.26 UniTraj-trained,
1.37 vanilla; caveat: those were TRAINED on nuScenes, ours is zero-shot from AV2).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from conformal import (SplitConformal, WeightedSplitConformal, NormalizedSplitConformal,  # noqa: E402
                       GroupConditionalConformal, domain_classifier_weights)
from attribution import fit_shift_regression, reweighting_removal_fraction  # noqa: E402
from shift_factors import FACTOR_NAMES  # noqa: E402

ROOT = SRC.parent


def load(tag, src):
    return dict(np.load(ROOT / "results" / "preds" / f"{tag}_from_{src}.npz", allow_pickle=True))


def minade_topk(d, k=5, step=5):
    """min over top-k-by-prob modes of ADE over every `step`-th horizon sample (2 Hz)."""
    P, pr, G, M = d["pred_trajs"], d["pred_probs"], d["gt"], d["gt_mask"]
    idx = np.arange(step - 1, P.shape[2], step)                      # t=0.5,1.0,...,6.0 s
    top = np.argsort(-pr, axis=1)[:, :k]
    out = []
    for i in range(len(P)):
        m = M[i, idx]
        if m.sum() == 0:
            continue
        e = np.linalg.norm(P[i, top[i]][:, idx] - G[i, idx][None], axis=-1)   # (k, T)
        out.append((e * m).sum(1).min() / m.sum())
    return float(np.mean(out)), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--same", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()

    cal, same, tgt = load(a.cal, a.src), load(a.same, a.src), load(a.target, a.src)
    out = {"source": a.src, "cal": a.cal, "same": a.same, "target": a.target,
           "n": {"cal": len(cal["scores"]), "same": len(same["scores"]), "target": len(tgt["scores"])}}

    # ---- base-model quality (context for the reader, not a pass/fail gate) ----
    ade5_t, nt = minade_topk(tgt, 5)
    ade5_s, ns_ = minade_topk(same, 5)
    out["base_model"] = {"target_minADE5_2Hz_zero_shot": ade5_t, "target_n": nt,
                         "same_domain_minADE5_2Hz": ade5_s, "same_n": ns_,
                         "note": "UniTraj paper nuScenes minADE5: AutoBot-UniTraj 1.26, AutoBot 1.37 "
                                 "(trained ON nuScenes; ours is zero-shot from the source dataset)"}

    # ---- H1: coverage per alpha; calibration = random 50% of cal, averaged over seeds ----
    h1 = {}
    for alpha in a.alphas:
        cov_same, cov_tgt, qh = [], [], []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(cal["scores"]))[: len(cal["scores"]) // 2]
            sc = SplitConformal(alpha).calibrate(cal["scores"][idx])
            cov_same.append(sc.evaluate(same["scores"])["coverage"])
            cov_tgt.append(sc.evaluate(tgt["scores"])["coverage"])
            qh.append(sc.q_hat_)
        n_t = len(tgt["scores"]); c = float(np.mean(cov_tgt)); se = float(np.sqrt(c * (1 - c) / n_t))
        h1[f"{alpha:.2f}"] = {
            "nominal": 1 - alpha, "q_hat_mean": float(np.mean(qh)),
            "same_domain_coverage": float(np.mean(cov_same)),
            "target_coverage": c, "target_gap": (1 - alpha) - c,
            "target_gap_95ci_binomial": [(1 - alpha) - c - 1.96 * se, (1 - alpha) - c + 1.96 * se],
            "target_gap_seed_range": [(1 - alpha) - max(cov_tgt), (1 - alpha) - min(cov_tgt)],
        }
    out["H1"] = h1

    # ---- shift factors + score summary ----
    Fs = np.vstack([cal["feats"], same["feats"]])
    out["shift_factor_means"] = {n: {"source": float(Fs[:, j].mean()), "target": float(tgt["feats"][:, j].mean())}
                                 for j, n in enumerate(FACTOR_NAMES)}
    out["score_mean"] = {"source": float(np.r_[cal["scores"], same["scores"]].mean()),
                         "target": float(tgt["scores"].mean())}

    # ---- H2: regression on UNIQUE pooled data (no duplicated files) ----
    S = np.r_[same["scores"], tgt["scores"]]; F = np.vstack([same["feats"], tgt["feats"]])
    reg = fit_shift_regression(S, F)
    out["H2_regression"] = {"pooled_adj_r2": reg["adj_r2"], "n": reg["n"], "coef": reg["coef_standardized"],
                            "within_source_adj_r2": fit_shift_regression(same["scores"], same["feats"][:, 1:])["adj_r2"],
                            "within_target_adj_r2": fit_shift_regression(tgt["scores"], tgt["feats"][:, 1:])["adj_r2"]}

    # ---- H3: recalibration methods on the target (alpha=0.10) ----
    alpha = 0.10
    res = {k: [] for k in ("uncorrected", "weighted", "normalized", "group")}
    radius = {k: [] for k in res}; removal = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(cal["scores"]))[: len(cal["scores"]) // 2]
        cs, cf, ts, tf = cal["scores"][idx], cal["feats"][idx], tgt["scores"], tgt["feats"]
        r = SplitConformal(alpha).calibrate(cs).evaluate(ts); res["uncorrected"].append(r["coverage"]); radius["uncorrected"].append(r["q_hat"])
        w = domain_classifier_weights(cf, tf, seed=seed)
        r = WeightedSplitConformal(alpha).calibrate(cs, w).evaluate(ts); res["weighted"].append(r["coverage"]); radius["weighted"].append(r["q_hat"])
        r = NormalizedSplitConformal(alpha).calibrate(cs, cf).evaluate(ts, tf); res["normalized"].append(r["coverage"]); radius["normalized"].append(r["q_hat_raw_equiv"])
        q1, q2 = np.percentile(cf[:, 1], [33, 66])
        r = GroupConditionalConformal(alpha).calibrate(cs, np.digitize(cf[:, 1], [q1, q2])).evaluate(ts, np.digitize(tf[:, 1], [q1, q2]))
        res["group"].append(r["coverage"]); radius["group"].append(r["q_hat"])
        removal.append(reweighting_removal_fraction(cs, cf, ts, tf, alpha, seed)["removal_fraction"])
    out["H3"] = {k: {"coverage_mean": float(np.mean(v)), "coverage_std": float(np.std(v)),
                     "gap": (1 - alpha) - float(np.mean(v)), "radius": float(np.mean(radius[k]))} for k, v in res.items()}
    out["H2_reweighting_removal_fraction"] = {"mean": float(np.mean(removal)), "min": float(np.min(removal)), "max": float(np.max(removal))}

    p = ROOT / "results" / f"pair_{a.src}__{a.target}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"wrote {p}")
    print(json.dumps({"base_model": out["base_model"], "H1_alpha0.10": out["H1"]["0.10"],
                      "H2_pooled_adj_r2": out["H2_regression"]["pooled_adj_r2"],
                      "H3": out["H3"], "removal": out["H2_reweighting_removal_fraction"]}, indent=1))


if __name__ == "__main__":
    main()
