"""
H2 — attribution of the cross-dataset coverage gap to the 7 pre-registered shift factors
(notes/falsification.md). Two deliverables:

  1. fit_shift_regression: adjusted R^2 of nonconformity score ~ shift factors, pooled
     across whatever datasets are given. Answers "how much of the SCORE variance do the
     7 factors explain" -- a necessary condition for them to explain the coverage GAP.

  2. reweighting_removal_fraction: for a given (source, target) pair, how much of the
     empirical coverage gap does an importance-weighted recalibration (using ONLY the 7
     factors, no labels) remove. This is the number the falsification note thresholds
     against: >= 40% removed supports H2, < 25% refutes it.

Consumes the same predict.py-format npz files as coverage_matrix.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from conformal import (  # noqa: E402
    SplitConformal, WeightedSplitConformal, domain_classifier_weights,
    split_conformal_quantile, coverage_and_efficiency,
)
from shift_factors import FACTOR_NAMES  # noqa: E402


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def fit_shift_regression(scores: np.ndarray, feats: np.ndarray, ridge: float = 1.0) -> dict:
    """
    Ridge regression of nonconformity score on standardized shift factors.
    Returns adjusted R^2, per-factor standardized coefficients, and the fit object's
    raw pieces (mean/std) so it can be reapplied.
    """
    n, d = feats.shape
    mu, sd = feats.mean(0), feats.std(0) + 1e-8
    Z = (feats - mu) / sd
    y = scores - scores.mean()

    A = Z.T @ Z + ridge * np.eye(d)
    w = np.linalg.solve(A, Z.T @ y)
    yhat = Z @ w
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum(y ** 2)) + 1e-12
    r2 = 1 - ss_res / ss_tot
    p = d  # regressors (ridge, so effective df is a bit less, but this is the standard report)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - p - 1, 1)

    return {
        "n": int(n),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "coef_standardized": {name: float(c) for name, c in zip(FACTOR_NAMES, w)},
    }


def reweighting_removal_fraction(cal_scores, cal_feats, target_scores, target_feats,
                                 alpha: float = 0.10, seed: int = 0) -> dict:
    """
    delta_before = nominal - coverage(uncorrected SCP calibrated on source, eval on target)
    delta_after  = nominal - coverage(importance-weighted SCP, same alpha)
    removal_fraction = (delta_before - delta_after) / delta_before   (clipped to [0,1]*10 for display)
    """
    nominal = 1 - alpha
    base = SplitConformal(alpha).calibrate(cal_scores).evaluate(target_scores)
    delta_before = nominal - base["coverage"]

    w = domain_classifier_weights(cal_feats, target_feats, seed=seed)
    weighted = WeightedSplitConformal(alpha).calibrate(cal_scores, w).evaluate(target_scores)
    delta_after = nominal - weighted["coverage"]

    if abs(delta_before) < 1e-6:
        frac = float("nan")
    else:
        frac = (delta_before - delta_after) / delta_before

    return {
        "alpha": alpha, "nominal": nominal,
        "delta_before": float(delta_before), "delta_after": float(delta_after),
        "removal_fraction": float(frac),
        "coverage_before": base["coverage"], "coverage_after": weighted["coverage"],
    }


def run(preds_dir: str, out_dir: str, alpha: float = 0.10):
    preds_dir = Path(preds_dir)
    files = {p.stem: p for p in preds_dir.glob("*.npz")}
    pairs = {}
    for stem in files:
        if "_from_" in stem:
            ev, src = stem.split("_from_")
            pairs[(ev, src)] = files[stem]
    sources = sorted({s for _, s in pairs})

    # (1) pooled regression across every available file
    all_scores, all_feats = [], []
    for p in files.values():
        d = _load(p)
        all_scores.append(d["scores"]); all_feats.append(d["feats"])
    all_scores = np.concatenate(all_scores); all_feats = np.concatenate(all_feats)
    reg = fit_shift_regression(all_scores, all_feats)
    print(f"[attribution] pooled regression: n={reg['n']}  adj_R2={reg['adj_r2']:.4f}")
    for name, c in reg["coef_standardized"].items():
        print(f"    {name:24s} {c:+.4f}")

    # (2) removal fraction per cross pair
    rows = []
    for src in sources:
        if (src, src) not in pairs:
            continue
        indom = _load(pairs[(src, src)])
        n = len(indom["scores"])
        rng = np.random.default_rng(0)
        cal_i = rng.choice(n, n // 2, replace=False)
        cs, cf = indom["scores"][cal_i], indom["feats"][cal_i]
        for ev, s in list(pairs):
            if s != src or ev == src:
                continue
            tgt = _load(pairs[(ev, src)])
            res = reweighting_removal_fraction(cs, cf, tgt["scores"], tgt["feats"], alpha)
            res.update(source=src, eval=ev)
            rows.append(res)
            print(f"  {src}->{ev}: delta {res['delta_before']:+.3f} -> {res['delta_after']:+.3f} "
                 f"(removed {res['removal_fraction']*100:.0f}%)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attribution_regression.json").write_text(json.dumps(reg, indent=2))
    (out_dir / "attribution_removal.json").write_text(json.dumps(rows, indent=2))
    if rows:
        mean_removal = float(np.nanmean([r["removal_fraction"] for r in rows]))
        print(f"\n[attribution] mean removal fraction across pairs: {mean_removal*100:.1f}%")
        print(f"  H2 thresholds: adj_R2>=0.35 (got {reg['adj_r2']:.3f}), "
             f"removal>=40% (got {mean_removal*100:.1f}%)")
    return reg, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(SRC.parent / "results" / "preds"))
    ap.add_argument("--out", default=str(SRC.parent / "results"))
    ap.add_argument("--alpha", type=float, default=0.10)
    a = ap.parse_args()
    run(a.preds, a.out, a.alpha)
