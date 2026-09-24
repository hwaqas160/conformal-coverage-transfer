"""
Unit checks of src/shift_inject.py transforms on a REAL AV2 batch (no model, no coverage -- so nothing about the
pre-registered outcomes is looked at): the perturbation must (a) leave what it should untouched, (b) change exactly
what it should, (c) do so at the measured strength.

  python src/tests/test_shift_inject.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["WANDB_MODE"] = "disabled"
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import json  # noqa: E402
from shift_inject import make_transform, KEYFRAMES, PARAMS  # noqa: E402
from unitraj_bridge import _load_cfg, build_loader  # noqa: E402


def main():
    params = json.loads(PARAMS.read_text())
    ds, ld = build_loader(str(SRC.parent / "data/av2_splits/val/test"), _load_cfg("autobot"), batch_size=32, num_workers=0)
    inp0 = next(iter(ld))["input_dict"]
    ok = True

    def clone():
        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp0.items()}

    # ---- hz2
    a = clone(); make_transform("hz2", params)(a)
    m = inp0["obj_trajs_mask"].bool()
    kf = list(KEYFRAMES)
    same_kf = torch.allclose(a["obj_trajs"][:, :, kf, :2], inp0["obj_trajs"][:, :, kf, :2])     # keyframes preserved
    full = m[:, :, kf].all(-1)
    mid = torch.tensor([i for i in range(21) if i not in kf])
    changed_mid = (a["obj_trajs"][:, :, mid, :2] - inp0["obj_trajs"][:, :, mid, :2]).abs().sum().item() > 0
    # interpolated samples lie exactly on the chord between keyframes
    p = a["obj_trajs"][..., :2]
    chord = p[:, :, 1, :] - (0.8 * p[:, :, 0, :] + 0.2 * p[:, :, 5, :])
    chord_ok = float(chord[full & m[:, :, 1]].abs().max()) < 1e-4
    untouched_other = torch.equal(a["obj_trajs"][..., 2:], inp0["obj_trajs"][..., 2:]) and torch.equal(a["map_polylines"], inp0["map_polylines"])
    print(f"hz2: keyframes preserved={same_kf}  interior changed={changed_mid}  on-chord={chord_ok}  other channels untouched={untouched_other}")
    ok &= same_kf and changed_mid and chord_ok and untouched_other

    # ---- map (lane dropout)
    b = clone(); make_transform("map", params, seed=1)(b)
    v0 = (inp0["map_polylines_mask"].sum(-1) > 0).float(); v1 = (b["map_polylines_mask"].sum(-1) > 0).float()
    kept = float(v1.sum() / v0.sum())
    whole = bool(((b["map_polylines_mask"] == 0) | (b["map_polylines_mask"] == inp0["map_polylines_mask"])).all())
    poly_level = bool(((b["map_polylines_mask"].sum(-1) == 0) | (b["map_polylines_mask"].sum(-1) == inp0["map_polylines_mask"].sum(-1))).all())
    print(f"map: fraction of valid polylines kept={kept:.3f} (target {params['lane_keep_prob']:.3f})  never adds points={whole}  whole-polyline={poly_level}")
    ok &= abs(kept - params["lane_keep_prob"]) < 0.04 and whole and poly_level and torch.equal(b["obj_trajs"], inp0["obj_trajs"])

    # ---- ground truth untouched by every transform
    gt_ok = all(torch.equal(x["center_gt_trajs"], inp0["center_gt_trajs"]) for x in (a, b))
    print(f"ground truth untouched by injections: {gt_ok}")
    ok &= gt_ok
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
