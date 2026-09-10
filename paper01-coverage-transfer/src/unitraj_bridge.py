"""
Thin bridge between the vendored UniTraj repo and Paper 01 code.

Responsibilities:
  * put UniTraj + its configs on the path
  * build an AutoBot(-Ego) model and load a Lightning checkpoint
  * build a dataset over ANY ScenarioNet-converted DB path
  * run inference and return plain numpy arrays (predictions, GT, masks, ids)

No conformal / experiment logic here -- see conformal.py, predict.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

# ------------------------------------------------------------------ paths / imports
REPO = Path(__file__).resolve().parents[1]
UNITRAJ = REPO / "code" / "UniTraj"
UNITRAJ_PKG = UNITRAJ / "unitraj"
for p in (str(UNITRAJ), str(UNITRAJ_PKG)):
    if p not in sys.path:
        sys.path.insert(0, p)

from omegaconf import OmegaConf  # noqa: E402


def _load_cfg(method: str = "autobot", overrides: dict | None = None):
    """Merge UniTraj's config.yaml + method yaml, apply overrides, return an open cfg."""
    base = OmegaConf.load(UNITRAJ_PKG / "configs" / "config.yaml")
    meth = OmegaConf.load(UNITRAJ_PKG / "configs" / "method" / f"{method}.yaml")
    cfg = OmegaConf.merge(base, meth)
    OmegaConf.set_struct(cfg, False)
    if overrides:
        for k, v in overrides.items():
            OmegaConf.update(cfg, k, v, merge=False)
    cfg.method = cfg  # UniTraj code reads cfg.method.<x> in places
    return cfg


def build_model(method: str = "autobot", ckpt_path: str | None = None,
                device: str = "cuda", cfg_overrides: dict | None = None):
    """Instantiate model, optionally load Lightning ckpt, set eval()."""
    # run from unitraj/ so the bare `from models import ...` inside the repo resolves
    cwd = os.getcwd()
    os.chdir(UNITRAJ_PKG)
    try:
        from unitraj.models import build_model as _bm
        cfg = _load_cfg(method, cfg_overrides)
        model = _bm(cfg)
        if ckpt_path:
            sd = torch.load(ckpt_path, map_location="cpu")
            sd = sd.get("state_dict", sd)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"[bridge] {len(missing)} missing keys (e.g. {missing[:3]})")
            if unexpected:
                print(f"[bridge] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
        model.eval().to(device)
        return model, cfg
    finally:
        os.chdir(cwd)


def build_loader(db_path: str, cfg, batch_size: int = 32, num_workers: int = 8,
                 max_data_num: int | None = None, is_validation: bool = True):
    """Dataset + DataLoader over a ScenarioNet-converted DB directory."""
    from torch.utils.data import DataLoader
    cwd = os.getcwd()
    os.chdir(UNITRAJ_PKG)
    try:
        from unitraj.datasets import build_dataset
        cfg.val_data_path = [str(db_path)]
        cfg.train_data_path = [str(db_path)]
        cfg.max_data_num = [max_data_num]
        cfg.starting_frame = [0]
        # absolute, stable cache so it survives changing CWD between runs
        cfg.cache_path = str(REPO / "data" / "unitraj_cache")
        ds = build_dataset(cfg, val=is_validation)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers, collate_fn=ds.collate_fn)
        return ds, loader
    finally:
        os.chdir(cwd)


@torch.no_grad()
def run_inference(model, loader, device: str = "cuda", limit_batches: int | None = None):
    """
    Returns a dict of numpy arrays aligned on axis 0 (one row per predicted agent):
      pred_trajs   (N, K, T, 2)
      pred_probs   (N, K)
      gt           (N, T, 2)
      gt_mask      (N, T)  bool
      final_idx    (N,)    int   last valid horizon index
      scenario_id  (N,)    str
      dataset_name (N,)    str
    plus 'raw_batches' generator hook is NOT kept (memory) -- use dump_predictions for features.
    """
    P_traj, P_prob, G, GM, FI, SID, DN = [], [], [], [], [], [], []
    for bi, batch in enumerate(loader):
        if limit_batches is not None and bi >= limit_batches:
            break
        _move(batch, device)
        out = model.predict(batch)
        pt = out["predicted_trajectory"].detach().cpu().numpy()   # (B,K,T,5)
        pp = out["predicted_probability"].detach().cpu().numpy()  # (B,K)
        inp = batch["input_dict"]
        gt = inp["center_gt_trajs"].detach().cpu().numpy()        # (B,T,4)
        gm = inp["center_gt_trajs_mask"].detach().cpu().numpy().astype(bool)  # (B,T)
        fi = inp["center_gt_final_valid_idx"].detach().cpu().numpy().astype(int)
        sid = np.asarray(inp["scenario_id"])
        dn = np.asarray(inp["dataset_name"])
        P_traj.append(pt[..., :2]); P_prob.append(pp)
        G.append(gt[..., :2]); GM.append(gm); FI.append(fi)
        SID.append(sid); DN.append(dn)

    return {
        "pred_trajs": np.concatenate(P_traj, 0),
        "pred_probs": np.concatenate(P_prob, 0),
        "gt": np.concatenate(G, 0),
        "gt_mask": np.concatenate(GM, 0),
        "final_idx": np.concatenate(FI, 0),
        "scenario_id": np.concatenate(SID, 0),
        "dataset_name": np.concatenate(DN, 0),
    }


def _move(batch, device):
    inp = batch["input_dict"]
    for k, v in inp.items() if False else list(inp.items()):
        if torch.is_tensor(v):
            inp[k] = v.to(device)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="ScenarioNet-converted DB dir")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=200, help="max scenes")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    m, cfg = build_model("autobot", a.ckpt, a.device)
    ds, dl = build_loader(a.db, cfg, batch_size=16, num_workers=4, max_data_num=a.n)
    print(f"dataset: {len(ds)} samples")
    res = run_inference(m, dl, a.device)
    for k, v in res.items():
        print(f"  {k:14s} {getattr(v, 'shape', None)} {v.dtype if hasattr(v,'dtype') else ''}")

    from conformal import nonconformity_scores, SplitConformal
    s = nonconformity_scores(res["pred_trajs"], res["gt"], res["gt_mask"])
    print(f"nonconformity: mean={s.mean():.2f} p50={np.median(s):.2f} p90={np.quantile(s,.9):.2f}")
