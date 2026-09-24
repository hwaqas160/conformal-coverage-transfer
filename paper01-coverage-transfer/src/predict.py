"""
Dump everything Paper 01's coverage experiment needs for ONE (checkpoint, dataset) pair:

  <out>/<tag>.npz  with arrays:
    pred_trajs   (N, K, T, 2)   float32
    pred_probs   (N, K)         float32
    gt           (N, T, 2)      float32
    gt_mask      (N, T)         bool
    final_idx    (N,)           int32
    scores       (N,)           float32   nonconformity (min-mode worst-step L2)
    feats        (N, 7)         float32   pre-registered shift factors
    scenario_id  (N,)           str
    dataset_name (N,)           str

Usage:
    python predict.py --ckpt <lightning.ckpt> --db <scenarionet_db> --dataset av2 \
                      --tag av2_from_av2 --out results/preds --n 20000 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unitraj_bridge import build_model, build_loader  # noqa: E402
from shift_factors import compute_factors, compute_factors_lf, FACTOR_NAMES, FACTOR_NAMES_LF  # noqa: E402
from conformal import nonconformity_scores  # noqa: E402


@torch.no_grad()
def dump(ckpt, db, dataset_key, tag, out_dir, n=None, device="cuda",
         batch_size=32, num_workers=8, method="autobot", out_file=None, limit_batches=None, transform=None):
    """transform: optional callable(input_dict_on_device) -> None, applied IN PLACE to the model inputs before
    prediction (used by shift_inject.py for controlled input perturbations).  Ground truth is never touched."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if ckpt:                       # build_model() chdirs into UniTraj; relative paths would break
        ckpt = str(Path(ckpt).resolve())
    if out_file:
        out_file = str(Path(out_file).resolve())
    model, cfg = build_model(method, ckpt, device)
    ds, loader = build_loader(db, cfg, batch_size=batch_size,
                              num_workers=num_workers, max_data_num=n)
    print(f"[predict] {tag}: {len(ds)} samples, ckpt={'none' if not ckpt else Path(ckpt).name}")

    PT, PP, G, GM, FI, FE, FL, PS, SID, DN, TT, KD = [], [], [], [], [], [], [], [], [], [], [], []
    for bi, batch in enumerate(loader):
        if limit_batches is not None and bi >= limit_batches:   # NB: --n is ignored for val-mode datasets
            break
        inp = batch["input_dict"]
        for k, v in list(inp.items()):
            if torch.is_tensor(v):
                inp[k] = v.to(device)
        if transform is not None:
            transform(inp)
        out = model.predict(batch)

        _full = out["predicted_trajectory"].detach().cpu().numpy()       # (B,K,T,5): x,y,bx,by,rho
        pt = _full[..., :2]
        ps = np.sqrt(_full[..., 2] ** 2 + _full[..., 3] ** 2).mean(axis=2)  # (B,K) mean-horizon Laplace scale
        pp = out["predicted_probability"].detach().cpu().numpy()
        bn = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
              for k, v in inp.items()}
        gt = bn["center_gt_trajs"][..., :2]
        gm = bn["center_gt_trajs_mask"].astype(bool)
        fi = bn["center_gt_final_valid_idx"].astype(np.int32)
        fe = compute_factors(bn, dataset_key)
        fl = compute_factors_lf(bn, dataset_key)
        sid = np.asarray(bn["scenario_id"]).astype("U160")   # was U40: truncated nuScenes ids (agent part lost)
        dn = np.asarray(bn["dataset_name"]).astype("U16")

        PT.append(pt); PP.append(pp); G.append(gt); GM.append(gm)
        FI.append(fi); FE.append(fe); FL.append(fl); PS.append(ps); SID.append(sid); DN.append(dn)
        # UniTraj's own scene labels (WOD trajectory taxonomy; Kalman FDE at 2/4/6 s) for conditional coverage
        TT.append(bn["trajectory_type"].astype(np.int8)); KD.append(bn["kalman_difficulty"].astype(np.float32))
        if bi % 50 == 0:
            print(f"  batch {bi}  ({sum(len(x) for x in PT)} rows, {time.time()-t0:.0f}s)")

    pred_trajs = np.concatenate(PT).astype(np.float32)
    gt = np.concatenate(G).astype(np.float32)
    gt_mask = np.concatenate(GM)
    scores = nonconformity_scores(pred_trajs, gt, gt_mask).astype(np.float32)

    path = Path(out_file) if out_file else Path(out_dir) / f"{tag}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_trajs=pred_trajs,
        pred_probs=np.concatenate(PP).astype(np.float32),
        gt=gt,
        gt_mask=gt_mask,
        final_idx=np.concatenate(FI),
        feats=np.concatenate(FE).astype(np.float32),
        feats_lf=np.concatenate(FL).astype(np.float32),
        pred_scale=np.concatenate(PS).astype(np.float32),
        scores=scores,
        scenario_id=np.concatenate(SID),
        dataset_name=np.concatenate(DN),
        factor_names=np.asarray(FACTOR_NAMES),
        factor_names_lf=np.asarray(FACTOR_NAMES_LF),
        trajectory_type=np.concatenate(TT),
        kalman_difficulty=np.concatenate(KD),
        schema=np.asarray(3),
    )
    print(f"[predict] wrote {path}  ({len(scores)} rows, {time.time()-t0:.0f}s)")
    print(f"          score mean={scores.mean():.2f} p50={np.median(scores):.2f} "
          f"p90={np.quantile(scores,.9):.2f}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--db", required=True)
    ap.add_argument("--dataset", required=True, help="key into shift_factors.DATASET_DT (av2/ns/wo/np)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=str(SRC.parent / "results" / "preds"))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit_batches", type=int, default=None, help="quick test: stop after N batches")
    ap.add_argument("--out_file", default=None, help="exact output .npz path (overrides --out/--tag)")
    a = ap.parse_args()
    dump(a.ckpt, a.db, a.dataset, a.tag, a.out, a.n, a.device, a.batch_size, a.num_workers, out_file=a.out_file, limit_batches=a.limit_batches)
