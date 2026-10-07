"""
Run UniTraj's own validation loop (same metric path as training, e.g. val/minADE6) for a checkpoint that was
never validated during training (e.g. the LAST epoch, when the final epoch doesn't land on check_val_every_n_epoch).
No training happens; this is read-only.

Usage: python src/validate_ckpt.py --ckpt <path> --val_db <db> --method wayformer [--val_subset 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from unitraj_bridge import build_model, UNITRAJ_PKG  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val_db", required=True)
    ap.add_argument("--method", default="wayformer")
    ap.add_argument("--cache_root", required=True)
    ap.add_argument("--val_subset", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import os
    import torch
    import numpy as np
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Subset

    ckpt = str(Path(a.ckpt).resolve())
    val_db = Path(a.val_db).resolve().as_posix()
    model, cfg = build_model(a.method, ckpt, "cuda")

    cwd = os.getcwd()
    os.chdir(UNITRAJ_PKG)
    try:
        from unitraj.datasets import build_dataset
        cfg.cache_path = str(Path(a.cache_root).resolve())
        cfg.val_data_path = [val_db]
        cfg.train_data_path = [val_db]
        cfg.max_data_num = [None]
        cfg.starting_frame = [0]
        cfg.method.eval_batch_size = a.batch_size
        val_set = build_dataset(cfg, val=True)
        if a.val_subset and len(val_set) > a.val_subset:
            idx = np.random.default_rng(a.seed).choice(len(val_set), a.val_subset, replace=False)
            val_set = Subset(val_set, idx.tolist())
            val_set.collate_fn = val_set.dataset.collate_fn if hasattr(val_set, "dataset") else val_set.collate_fn
        val_loader = DataLoader(val_set, batch_size=a.batch_size, shuffle=False,
                                num_workers=a.num_workers, collate_fn=val_set.collate_fn if hasattr(val_set, "collate_fn")
                                else val_set.dataset.collate_fn)
        print(f"[validate_ckpt] {a.ckpt}: val={len(val_set)} samples")
    finally:
        os.chdir(cwd)

    trainer = pl.Trainer(devices=1, accelerator="cuda", logger=False, enable_checkpointing=False)
    out = trainer.validate(model, dataloaders=val_loader, verbose=True)
    print("[validate_ckpt] result:", out)
