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
from unitraj_bridge import UNITRAJ_PKG, _load_cfg, DEFAULT_CACHE, cache_guard, cache_mark  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_db", required=True, nargs="+",
                    help="one or more ScenarioNet DB dirs, e.g. the 11 chunked nuscenes_scenarionet/train_cNN dirs")
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
    ap.add_argument("--resume", default=None, help="full Lightning resume (optimizer+epoch); wins over --init_ckpt")
    ap.add_argument("--init_ckpt", default=None, help="initialise WEIGHTS ONLY from this ckpt (fine-tuning)")
    ap.add_argument("--lr_sched", type=int, nargs="+", default=None,
                    help="MultiStepLR milestones in epochs (gamma=0.5 inside UniTraj). Default = UniTraj's "
                         "[10,20,30,40,50], i.e. NO decay inside a <=10-epoch run")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads, CPU only")
    ap.add_argument("--method", default="autobot", help="UniTraj method config: autobot | wayformer")
    ap.add_argument("--profile_steps", type=int, default=0,
                    help="diagnostic: run N train batches with Lightning's SimpleProfiler, print breakdown, exit")
    ap.add_argument("--cache_root", default=None,
                    help="UniTraj cache root (default data/unitraj_cache on F:). Use an SSD copy for shuffled "
                         "full-data training: on the F: HDD one random-access step took 262 s.")
    a = ap.parse_args()
    # resolve to absolute BEFORE any chdir
    a.train_db = [Path(p).resolve().as_posix() for p in a.train_db]
    a.val_db = Path(a.val_db).resolve().as_posix()
    if a.resume:
        a.resume = Path(a.resume).resolve().as_posix()
    if a.init_ckpt:
        a.init_ckpt = Path(a.init_ckpt).resolve().as_posix()

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
    cfg = _load_cfg(a.method)
    cfg.exp_name = a.exp
    cfg.seed = a.seed
    cfg.debug = False
    cfg.load_num_workers = a.num_workers
    cache_root = Path(a.cache_root).resolve() if a.cache_root else DEFAULT_CACHE
    cache_guard(a.train_db + [a.val_db], cache_root)       # refuse silent cross-DB cache reuse
    cfg.cache_path = str(cache_root)
    cfg.train_data_path = a.train_db
    cfg.val_data_path = [a.val_db]
    cfg.max_data_num = [a.limit_train] * len(a.train_db)
    cfg.starting_frame = [0] * len(a.train_db)
    cfg.method.train_batch_size = a.batch
    cfg.method.eval_batch_size = max(a.batch * 2, 16)
    cfg.method.max_epochs = a.epochs
    # NOTE: UniTraj's optimizer reads config['learning_rate'] / ['learning_rate_sched'] from the TOP
    # level. _load_cfg() also stores a nested copy at cfg.method, so writing only cfg.method.* is a
    # silent no-op (found 2026-09-21: log kept printing lr 7.5e-4 for --lr 3.75e-4). Set both.
    cfg.method.learning_rate = a.lr
    cfg.learning_rate = a.lr
    if a.lr_sched:
        cfg.method.learning_rate_sched = a.lr_sched
        cfg.learning_rate_sched = a.lr_sched

    model = build_model(cfg)
    if a.init_ckpt and not a.resume:
        _sd = torch.load(Path(a.init_ckpt).resolve().as_posix(), map_location="cpu")
        _sd = _sd.get("state_dict", _sd)
        _miss, _unexp = model.load_state_dict(_sd, strict=False)
        print(f"[train] initialised weights from {a.init_ckpt} (missing={len(_miss)}, unexpected={len(_unexp)})")
    train_set = build_dataset(cfg, val=False)
    val_set = build_dataset(cfg, val=True)
    cache_mark(a.train_db + [a.val_db], cache_root)
    if a.val_subset and len(val_set) > a.val_subset:
        from torch.utils.data import Subset
        import numpy as _np
        idx = _np.random.default_rng(0).choice(len(val_set), a.val_subset, replace=False)
        val_set = Subset(val_set, idx.tolist())
        val_set.collate_fn = train_set.collate_fn
    print(f"[train] train={len(train_set)}  val={len(val_set)}")

    # pin_memory: page-locked batches make the host->device copy asynchronous and fast (py-spy showed 44% of the
    # main thread inside the blocking pageable-memory transfer).  Purely a transfer-path change.
    pin = a.device == "cuda"
    train_loader = DataLoader(train_set, batch_size=a.batch, shuffle=True, drop_last=True,
                              num_workers=a.num_workers, collate_fn=train_set.collate_fn,
                              persistent_workers=a.num_workers > 0, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=cfg.method.eval_batch_size, shuffle=False,
                            num_workers=a.num_workers, collate_fn=val_set.collate_fn,
                            persistent_workers=a.num_workers > 0, pin_memory=pin)

    ckpt_cb = ModelCheckpoint(
        monitor="val/minADE6", mode="min", save_top_k=2, save_last=True,
        filename="epoch{epoch:02d}-minADE{val/minADE6:.3f}", auto_insert_metric_name=False,
        dirpath=str(SRC.parent / "results" / "ckpts" / a.exp),
    )
    # mid-epoch safety net: a crash used to lose an entire epoch (e.g. a DataLoader
    # worker dying 32% through epoch 0, discovered a day later with zero checkpoints
    # to resume from). This saves "last.ckpt" every 200 steps regardless of val/epoch
    # boundaries, so --resume never loses more than a few minutes of CPU-bound training.
    step_ckpt_cb = ModelCheckpoint(
        dirpath=str(SRC.parent / "results" / "ckpts" / a.exp),
        filename="last", save_last=True, every_n_train_steps=200,
        save_top_k=0,  # this callback only maintains "last.ckpt"; ckpt_cb tracks the best
    )
    trainer = pl.Trainer(
        max_epochs=a.epochs,
        devices=1, accelerator=a.device, strategy="auto",
        precision=32,
        accumulate_grad_batches=a.accum,
        gradient_clip_val=cfg.method.grad_clip_norm,
        check_val_every_n_epoch=a.val_every,
        callbacks=[ckpt_cb, step_ckpt_cb, LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger(str(SRC.parent / "results" / "logs"), name=a.exp),
        log_every_n_steps=50,
        enable_progress_bar=True,
        **({"profiler": "simple", "limit_train_batches": a.profile_steps, "limit_val_batches": 0}
           if a.profile_steps else {}),
    )
    if a.profile_steps:
        trainer.callbacks = [c for c in trainer.callbacks if not isinstance(c, ModelCheckpoint)]
        trainer.fit(model, train_loader)
        return
    trainer.fit(model, train_loader, val_loader, ckpt_path=a.resume)
    print(f"[train] best: {ckpt_cb.best_model_path}  ({ckpt_cb.best_model_score})")


if __name__ == "__main__":
    main()
