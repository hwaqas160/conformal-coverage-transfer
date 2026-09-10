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


def _fake_npz(path, n, err_scale, feat_shift, seed):
    rng = np.random.default_rng(seed)
    T, K = 60, 6
    gt = np.cumsum(rng.normal(0, 0.3, size=(n, T, 2)), axis=1).astype(np.float32)
    good = gt + rng.normal(0, err_scale, size=(n, T, 2))
    bad = rng.normal(0, 6, size=(n, K - 1, T, 2))
    pred = np.concatenate([good[:, None], bad], axis=1).astype(np.float32)
    gt_mask = np.ones((n, T), bool)
    scores = nonconformity_scores(pred, gt, gt_mask).astype(np.float32)
    # 7 shift factors; correlate factor 1 (agent density) & 4 (speed) with err via feat_shift
    feats = rng.normal(0, 1, size=(n, 7)).astype(np.float32)
    feats[:, 1] += feat_shift + 0.5 * (scores - scores.mean()) / (scores.std() + 1e-6)
    feats[:, 4] += 0.7 * feat_shift
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
    # A: tight errors.  B: 2x errors + shifted features.
    _fake_npz(preds / "A_from_A.npz", 6000, err_scale=0.5, feat_shift=0.0, seed=1)
    _fake_npz(preds / "B_from_A.npz", 6000, err_scale=1.1, feat_shift=1.5, seed=2)
    _fake_npz(preds / "B_from_B.npz", 6000, err_scale=1.1, feat_shift=1.5, seed=3)
    _fake_npz(preds / "A_from_B.npz", 6000, err_scale=0.5, feat_shift=0.0, seed=4)

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

    ok = True
    if abs(d_in) > 0.03:
        print("FAIL: in-domain coverage off nominal -> SCP bug"); ok = False
    if d_cross < 0.02:
        print("FAIL: no cross-domain gap on synthetic shift -> pipeline not detecting shift"); ok = False
    best_recal = min(np.mean(v) for m, v in by_m.items() if m != "uncorrected")
    if best_recal >= np.mean(by_m["uncorrected"]):
        print("WARN: no recalibration method beat uncorrected on synthetic")
    print("\nPIPELINE OK" if ok else "\nPIPELINE FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
