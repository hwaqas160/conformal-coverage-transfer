"""Which of the 21 history frames are valid for the TARGET agent, per dataset? (H21 design check.)
Reads a few batches via the project's loader, no model. Usage: python src/inspect_history_mask.py"""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from unitraj_bridge import _load_cfg, build_loader  # noqa: E402

SETS = {"av2test": ("data/av2_splits/val/test", "C:\\p01_cache"),
        "wmtest": ("data/waymo_splits/wmscene/wmtest", "C:\\p01_cache_wf"),
        "ns": ("data/nuscenes_scenarionet/val", None)}

if __name__ == "__main__":
    for name, (db, cache) in SETS.items():
        method = "wayformer" if cache and "wf" in cache else "autobot"
        if name == "ns":
            continue
        cfg = _load_cfg(method)
        root = Path(__file__).resolve().parent.parent
        ds, ld = build_loader(str(root / db), cfg, batch_size=32, num_workers=0, cache_root=cache)
        valid_counts, lead = [], np.zeros(21)
        for bi, b in enumerate(ld):
            if bi >= 6:
                break
            m = b["input_dict"]["obj_trajs_mask"].numpy()          # (B, A, 21)
            ctr = m[:, 0]                                          # target agent is slot 0 in UniTraj's layout
            valid_counts += ctr.sum(-1).tolist()
            lead += ctr.sum(0)
        valid_counts = np.array(valid_counts)
        print(f"{name:8s} ({method}) target-agent valid history frames: mean {valid_counts.mean():.1f}, "
              f"min {valid_counts.min():.0f}, max {valid_counts.max():.0f}; per-frame valid share (t=-2.0..0 s):")
        print("   ", np.round(lead / lead.max(), 2).tolist())
