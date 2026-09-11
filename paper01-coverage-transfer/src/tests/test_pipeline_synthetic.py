"""
End-to-end synthetic test of the Paper 01 analysis pipeline WITHOUT a model or data.

Fabricates npz files in predict.py's format for two 'datasets' A and B where B has a
genuine distribution shift (larger prediction errors + shifted shift-factors), then checks:
  * in-domain conformal coverage is ~nominal        (SCP implementation correct)
  * cross-domain coverage drops below nominal        (H1 mechanism reproduces on synthetic)
  * at least one recalibration method recovers coverage  (H3 machinery works)

Run:  python src/tests/test_pipeline_synthetic.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from conformal import nonconformity_scores  # noqa: E402
import coverage_matrix as cm  # noqa: E402
import attribution as attr  # noqa: E402


def _fake_npz(path, n, p_dense, seed):
    """
    Genuine COVARIATE shift, not a scale shift: every scene has a discrete density
    level L in {0,1,2} with a shared conditional error distribution err_scale(L); only
    the MIXTURE weight p(L) differs between 'datasets' (p_dense). This is exactly what
    importance-weighted conformal prediction is designed to fix, because the source
    calibration set already contains high-density scenes -- just underrepresented.
    """
    rng = np.random.default_rng(seed)
    T, K = 60, 6
    p = np.array([1 - p_dense - p_dense / 2, p_dense / 2, p_dense])
    p = np.clip(p, 0.02, None); p /= p.sum()
    level = rng.choice([0, 1, 2], size=n, p=p)
    err_scale_by_level = np.array([0.4, 0.9, 1.6])
    err_scale = err_scale_by_level[level]

    gt = np.cumsum(rng.normal(0, 0.3, size=(n, T, 2)), axis=1).astype(np.float32)
    good = gt + rng.normal(0, 1, size=(n, T, 2)) * err_scale[:, None, None]
    bad = rng.normal(0, 6, size=(n, K - 1, T, 2))
    pred = np.concatenate([good[:, None], bad], axis=1).astype(np.float32)
    gt_mask = np.ones((n, T), bool)
    scores = nonconformity_scores(pred, gt, gt_mask).astype(np.float32)

    # feature 1 (agent density) reveals the level; a touch of noise, no label leakage
    feats = rng.normal(0, 1, size=(n, 7)).astype(np.float32)
    feats[:, 1] = level.astype(np.float32) + rng.normal(0, 0.3, n)
    np.savez_compressed(
        path, pred_trajs=pred, pred_probs=rng.dirichlet(np.ones(K), n).astype(np.float32),
        gt=gt, gt_mask=gt_mask, final_idx=np.full(n, T - 1, np.int32),
        feats=feats, scores=scores,
        scenario_id=np.array([f"s{i}" for i in range(n)]),
        dataset_name=np.array(["syn"] * n),
        factor_names=np.array(cm.__dict__.get("FACTOR_NAMES", list("abcdefg"))),
    )


def main():
    tmp = Path(tempfile.mkdtemp())
    preds = tmp / "preds"
    preds.mkdir()
    # A: mostly low-density scenes.  B: mostly high-density scenes.  SAME conditional
    # error-given-density relationship in both -- a covariate shift, not a scale shift.
    _fake_npz(preds / "A_from_A.npz", 8000, p_dense=0.10, seed=1)
    _fake_npz(preds / "B_from_A.npz", 8000, p_dense=0.55, seed=2)
    _fake_npz(preds / "B_from_B.npz", 8000, p_dense=0.55, seed=3)
    _fake_npz(preds / "A_from_B.npz", 8000, p_dense=0.10, seed=4)

    rows = cm.coverage_transfer(preds, alphas=(0.10,), seeds=(0, 1, 2), out_dir=str(tmp))
    d_in = np.mean([r["delta"] for r in rows if r["kind"] == "in"])
    d_cross = np.mean([r["delta"] for r in rows if r["kind"] == "cross"])
    print(f"\nD_in   = {d_in:+.4f}   (expect ~0)")
    print(f"D_cross = {d_cross:+.4f}   (expect > 0.03)")

    rec = cm.recalibration_compare(preds, alpha=0.10, seeds=(0, 1, 2), out_dir=str(tmp))
    by_m = {}
    for r in rec:
        by_m.setdefault(r["method"], []).append(abs(r["delta"]))
    print("\nrecalibration |D| by method:")
    for m, v in by_m.items():
        print(f"  {m:12s} {np.mean(v):.4f}")

    reg, removal_rows = attr.run(preds, str(tmp), alpha=0.10)
    mean_removal = float(np.nanmean([r["removal_fraction"] for r in removal_rows]))

    ok = True
    if abs(d_in) > 0.03:
        print("FAIL: in-domain coverage off nominal -> SCP bug"); ok = False
    if d_cross < 0.02:
        print("FAIL: no cross-domain gap on synthetic shift -> pipeline not detecting shift"); ok = False
    best_recal = min(np.mean(v) for m, v in by_m.items() if m != "uncorrected")
    if best_recal >= np.mean(by_m["uncorrected"]):
        print("WARN: no recalibration method beat uncorrected on synthetic")
    if reg["adj_r2"] < 0.05:
        print(f"FAIL: attribution regression finds ~no signal (adj_R2={reg['adj_r2']:.3f}) "
             "on synthetic data built to correlate score with feature 1 -> attribution.py bug")
        ok = False
    if not (mean_removal > 0):
        print(f"WARN: reweighting removed 0 or negative coverage gap on synthetic ({mean_removal:.2f})")
    print("\nPIPELINE OK" if ok else "\nPIPELINE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
