"""
Train AutoBot-Ego with UniTraj on a ScenarioNet DB, tuned for a single Quadro P2000.

Thin wrapper around UniTraj's Lightning model so we control:
  * the train / val DB paths (scene-level split done upstream via split_db.py)
  * P2000-safe trainer settings (single device, no DDP, bf16 off, grad accum)
  * checkpoint on val minADE6

Usage:
  python train_autobot.py --train_db <db> --val_db <db> --exp av2 \
      --epochs 40 --batch 16 --accum 2 --limit_train 120000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from unitraj_bridge import UNITRAJ_PKG, _load_cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_db", required=True)
    ap.add_argument("--val_db", required=True)
    ap.add_argument("--exp", required=True, help="experiment name (ckpt dir)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)   # P2000: 16.3 samp/s, 2.5GB; 48 regresses
    ap.add_argument("--accum", type=int, default=1, help="grad accumulation")
    ap.add_argument("--val_every", type=int, default=2, help="check_val_every_n_epoch")
    ap.add_argument("--val_subset", type=int, default=None, help="cap val scenes for speed")
    ap.add_argument("--lr", type=float, default=7.5e-4)
    ap.add_argument("--limit_train", type=int, default=None, help="cap # train scenes")
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads, CPU only")
    a = ap.parse_args()
    # resolve to absolute BEFORE any chdir
    a.train_db = Path(a.train_db).resolve().as_posix()
    a.val_db = Path(a.val_db).resolve().as_posix()
    if a.resume:
        a.resume = Path(a.resume).resolve().as_posix()

    os.environ.setdefault("WANDB_MODE", "disabled")
    torch.set_float32_matmul_precision("medium")
    if a.device == "cpu" and a.threads:
        torch.set_num_threads(a.threads)

    os.chdir(UNITRAJ_PKG)
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from unitraj.models import build_model
    from unitraj.datasets import build_dataset
    from unitraj.utils.utils import set_seed

    set_seed(a.seed)
    cfg = _load_cfg("autobot")
    cfg.exp_name = a.exp
    cfg.seed = a.seed
    cfg.debug = False
    cfg.load_num_workers = a.num_workers
    cfg.cache_path = str(SRC.parent / "data" / "unitraj_cache")
    cfg.train_data_path = [a.train_db]
    cfg.val_data_path = [a.val_db]
    cfg.max_data_num = [a.limit_train]
    cfg.starting_frame = [0]
    cfg.method.train_batch_size = a.batch
    cfg.method.eval_batch_size = max(a.batch * 2, 16)
    cfg.method.max_epochs = a.epochs
    cfg.method.learning_rate = a.lr

    model = build_model(cfg)
    train_set = build_dataset(cfg, val=False)
    val_set = build_dataset(cfg, val=True)
    if a.val_subset and len(val_set) > a.val_subset:
        from torch.utils.data import Subset
        import numpy as _np
        idx = _np.random.default_rng(0).choice(len(val_set), a.val_subset, replace=False)
        val_set = Subset(val_set, idx.tolist())
        val_set.collate_fn = train_set.collate_fn
    print(f"[train] train={len(train_set)}  val={len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=a.batch, shuffle=True, drop_last=True,
                              num_workers=a.num_workers, collate_fn=train_set.collate_fn,
                              persistent_workers=a.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=cfg.method.eval_batch_size, shuffle=False,
                            num_workers=a.num_workers, collate_fn=val_set.collate_fn,
                            persistent_workers=a.num_workers > 0)

    ckpt_cb = ModelCheckpoint(
        monitor="val/minADE6", mode="min", save_top_k=2, save_last=True,
        filename="epoch{epoch:02d}-minADE{val/minADE6:.3f}", auto_insert_metric_name=False,
        dirpath=str(SRC.parent / "results" / "ckpts" / a.exp),
    )
    trainer = pl.Trainer(
        max_epochs=a.epochs,
        devices=1, accelerator=a.device, strategy="auto",
        precision=32,
        accumulate_grad_batches=a.accum,
        gradient_clip_val=cfg.method.grad_clip_norm,
        check_val_every_n_epoch=a.val_every,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger(str(SRC.parent / "results" / "logs"), name=a.exp),
        log_every_n_steps=50,
        enable_progress_bar=True,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=a.resume)
    print(f"[train] best: {ckpt_cb.best_model_path}  ({ckpt_cb.best_model_score})")


if __name__ == "__main__":
    main()
