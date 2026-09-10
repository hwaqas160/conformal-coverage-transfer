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
from shift_factors import compute_factors, FACTOR_NAMES  # noqa: E402
from conformal import nonconformity_scores  # noqa: E402


@torch.no_grad()
def dump(ckpt, db, dataset_key, tag, out_dir, n=None, device="cuda",
         batch_size=32, num_workers=8, method="autobot"):
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    model, cfg = build_model(method, ckpt, device)
    ds, loader = build_loader(db, cfg, batch_size=batch_size,
                              num_workers=num_workers, max_data_num=n)
    print(f"[predict] {tag}: {len(ds)} samples, ckpt={'none' if not ckpt else Path(ckpt).name}")

    PT, PP, G, GM, FI, FE, SID, DN = [], [], [], [], [], [], [], []
    for bi, batch in enumerate(loader):
        inp = batch["input_dict"]
        for k, v in list(inp.items()):
            if torch.is_tensor(v):
                inp[k] = v.to(device)
        out = model.predict(batch)

        pt = out["predicted_trajectory"].detach().cpu().numpy()[..., :2]
        pp = out["predicted_probability"].detach().cpu().numpy()
        bn = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
              for k, v in inp.items()}
        gt = bn["center_gt_trajs"][..., :2]
        gm = bn["center_gt_trajs_mask"].astype(bool)
        fi = bn["center_gt_final_valid_idx"].astype(np.int32)
        fe = compute_factors(bn, dataset_key)
        sid = np.asarray(bn["scenario_id"]).astype("U40")
        dn = np.asarray(bn["dataset_name"]).astype("U16")

        PT.append(pt); PP.append(pp); G.append(gt); GM.append(gm)
        FI.append(fi); FE.append(fe); SID.append(sid); DN.append(dn)
        if bi % 50 == 0:
            print(f"  batch {bi}  ({sum(len(x) for x in PT)} rows, {time.time()-t0:.0f}s)")

    pred_trajs = np.concatenate(PT).astype(np.float32)
    gt = np.concatenate(G).astype(np.float32)
    gt_mask = np.concatenate(GM)
    scores = nonconformity_scores(pred_trajs, gt, gt_mask).astype(np.float32)

    path = Path(out_dir) / f"{tag}.npz"
    np.savez_compressed(
        path,
        pred_trajs=pred_trajs,
        pred_probs=np.concatenate(PP).astype(np.float32),
        gt=gt,
        gt_mask=gt_mask,
        final_idx=np.concatenate(FI),
        feats=np.concatenate(FE).astype(np.float32),
        scores=scores,
        scenario_id=np.concatenate(SID),
        dataset_name=np.concatenate(DN),
        factor_names=np.asarray(FACTOR_NAMES),
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
    a = ap.parse_args()
    dump(a.ckpt, a.db, a.dataset, a.tag, a.out, a.n, a.device, a.batch_size, a.num_workers)
