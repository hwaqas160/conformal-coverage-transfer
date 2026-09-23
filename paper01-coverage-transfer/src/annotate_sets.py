"""
Model-independent scene annotations for conditional-coverage analysis (Addendum B7).

Trajectory type and Kalman difficulty are properties of a scene's ground truth, not of any model, so they are
extracted once per evaluation set -- straight from UniTraj's own batch fields, i.e. UniTraj's exact definitions
(WOD trajectory taxonomy; Kalman-filter FDE at 2/4/6 s) -- and joined to any model's predictions by scenario_id.

  results/preds2/_annot/<set>.npz : scenario_id (N,), trajectory_type (N,) int, kalman_difficulty (N,3),
                                    speed_now (N,) m/s of the predicted agent at t=0

  python src/annotate_sets.py                 # all sets in predict_all.SETS, skips existing
  python src/annotate_sets.py --sets av2test ns
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402
from predict_all import SETS  # noqa: E402

OUT = ROOT / "results" / "preds2" / "_annot"
TRAJ_TYPES = ["stationary", "straight", "straight_right", "straight_left",
              "right_u_turn", "right_turn", "left_u_turn", "left_turn"]


def gt_keys(gt_xy: np.ndarray) -> np.ndarray:
    """Per-row content key: hash of the ground-truth future rounded to 1 cm.  Lets annotations be joined to
    prediction files that stored scenario ids truncated to 40 chars (schema <= 3), where nuScenes ids collide."""
    import hashlib
    r = np.round(np.asarray(gt_xy, np.float64), 2)
    return np.asarray([hashlib.sha1(r[i].tobytes()).hexdigest()[:16] for i in range(len(r))])


def annotate(name: str, num_workers: int = 4) -> Path:
    from unitraj_bridge import _load_cfg, build_loader
    db, _ = SETS[name]
    cfg = _load_cfg("autobot")
    ds, loader = build_loader(str(ROOT / db), cfg, batch_size=64, num_workers=num_workers)
    SID, TT, KD, SP, GK = [], [], [], [], []
    t0 = time.time()
    for batch in loader:
        b = batch["input_dict"]
        SID.append(np.asarray(b["scenario_id"]).astype("U160"))       # full length: nuScenes ids exceed 40 chars
        GK.append(gt_keys(b["center_gt_trajs"].numpy()[..., :2]))
        TT.append(b["trajectory_type"].numpy().astype(np.int8))
        KD.append(b["kalman_difficulty"].numpy().astype(np.float32))
        # predicted agent's velocity at the current step (obj_trajs features 35:37 = vx, vy; see shift_factors.py)
        ti = b["track_index_to_predict"].numpy()
        ot = b["obj_trajs"].numpy()
        v = ot[np.arange(len(ti)), ti, -1, 35:37]
        SP.append(np.hypot(v[:, 0], v[:, 1]).astype(np.float32))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.npz"
    np.savez_compressed(p, scenario_id=np.concatenate(SID), trajectory_type=np.concatenate(TT),
                        kalman_difficulty=np.concatenate(KD), speed_now=np.concatenate(SP),
                        gt_key=np.concatenate(GK), traj_type_names=np.asarray(TRAJ_TYPES))
    journal.event(f"annotate_{name}", "artifact", f"{p.relative_to(ROOT)} ({sum(len(s) for s in SID)} rows, {time.time()-t0:.0f}s)")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()
    for s in a.sets:
        if (OUT / f"{s}.npz").exists():
            print(f"[skip] {s}"); continue
        print(f"[annotate] {s} -> {annotate(s, a.num_workers)}")


if __name__ == "__main__":
    main()
