"""
Addendum B4 -- controlled shift injection: which measurable AV2-vs-nuScenes difference reproduces the coverage loss?

Each injection perturbs ONE aspect of the model inputs of AV2 test scenes (ground truth untouched) and the
resulting coverage of the AV2-calibrated threshold is compared with the real AV2->nuScenes gap.

  hz2      history re-sampled to nuScenes' native 2 Hz: keep the 5 keyframes (t = -20,-15,-10,-5,0 of the 21-step
           10 Hz history) and linearly interpolate in between            (all agents)
  jitter   i.i.d. Gaussian noise on every valid past position, sigma from `estimate_noise_sigma` (data-driven:
           excess 2-Hz quadratic-fit residual of nuScenes over AV2; NOT guessed)
  map      lane dropout: each map POLYLINE (lane segment) dropped with prob 1 - p, p = nuScenes/AV2 median of the
           label-free factor n_lanes_near_ego (measured: 52 vs 85 -> p ~ 0.61).  [Replaces the registered per-point
           thinning: measured per-scene valid map-point counts are equal (2316 vs 2350), so that would be vacuous.]
  (comp)   scene-composition reweighting (covariate shift by construction) needs no inference; it is computed in
           analyze_addB from existing predictions.

AutoBot consumes only obj_trajs[..., :2] + obj_trajs_mask and map_polylines[..., :2] + map_polylines_mask, so
these three perturbations are exact for it.  (Wayformer also reads velocity/heading channels -> its own handling.)

  python src/shift_inject.py sigma                              # noise level -> results/addB/inject_params.json
  python src/shift_inject.py run --model av2_cpu_v2 --ckpt auto --conds hz2 map hz2map [--n_batches 20] [--device cpu]
Outputs  results/preds2/<model>/inj_<cond>.npz  (same schema as predict.py)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402

PARAMS = ROOT / "results" / "addB" / "inject_params.json"
KEYFRAMES = (0, 5, 10, 15, 20)          # of the 21-step history: every 5th step = the 2 Hz samples (t = -2.0 ... 0 s)


# ------------------------------------------------------------------------------------------ noise estimate
def _quad_resid_rms(P: np.ndarray) -> np.ndarray:
    """P (M, 5, 2) positions at the 5 keyframes -> per-agent unbiased noise variance from a quadratic fit
    (3 parameters, 5 points => 2 dof per coordinate)."""
    t = np.arange(-4, 1, dtype=np.float64)
    X = np.stack([np.ones_like(t), t, t * t], 1)                       # (5, 3)
    H = X @ np.linalg.pinv(X)                                          # hat matrix
    R = P - np.einsum("ij,mjc->mic", H, P)                             # residuals
    return (R ** 2).sum(axis=(1, 2)) / (2 * 2)                         # sum of squares / (dof * coords)


def estimate_noise_sigma(db: str, n_batches: int = 60, seed: int = 0) -> dict:
    from unitraj_bridge import _load_cfg, build_loader
    cfg = _load_cfg("autobot")
    ds, ld = build_loader(str(ROOT / db), cfg, batch_size=32, num_workers=3)
    var, dens, n_agents = [], [], 0
    for bi, batch in enumerate(ld):
        if bi >= n_batches:
            break
        b = batch["input_dict"]
        pos = b["obj_trajs"][..., :2].numpy().astype(np.float64)       # (B, A, 21, 2)
        msk = b["obj_trajs_mask"].numpy().astype(bool)                 # (B, A, 21)
        full = msk.all(-1)                                             # complete history only
        # moving agents only (a parked car's residual is pure noise but its motion prior differs)
        span = np.linalg.norm(pos[..., -1, :] - pos[..., 0, :], axis=-1)
        sel = full & (span > 2.0)
        P = pos[sel][:, list(KEYFRAMES), :]
        if len(P):
            var.append(_quad_resid_rms(P)); n_agents += len(P)
        mp = b["map_polylines_mask"].numpy().astype(bool)
        dens.append(mp.sum(axis=(1, 2)))                               # valid lane points per scene
    v = np.concatenate(var)
    return {"sigma_m": float(np.sqrt(np.mean(v))), "sigma_m_median_agent": float(np.sqrt(np.median(v))),
            "n_agents": int(n_agents), "map_points_per_scene_median": float(np.median(np.concatenate(dens)))}


def cmd_sigma(a):
    out = {}
    for name, db in (("av2", "data/av2_splits/val/test"), ("ns", "data/nuscenes_scenarionet/val")):
        out[name] = estimate_noise_sigma(db, n_batches=a.n_batches)
        print(name, out[name])
    s_ns, s_av = out["ns"]["sigma_m"], out["av2"]["sigma_m"]
    out["jitter_sigma_m"] = float(np.sqrt(max(s_ns ** 2 - s_av ** 2, 0.0)))
    fl = {k: np.load(ROOT / "results" / "preds2" / "av2_cpu_v2" / f"{k}.npz", allow_pickle=True) for k in ("av2test", "ns")}
    li = list(fl["ns"]["factor_names_lf"]).index("n_lanes_near_ego")
    out["lanes_median"] = {"av2": float(np.median(fl["av2test"]["feats_lf"][:, li])), "ns": float(np.median(fl["ns"]["feats_lf"][:, li]))}
    out["lane_keep_prob"] = float(min(1.0, out["lanes_median"]["ns"] / out["lanes_median"]["av2"]))
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(out, indent=1))
    print(f"jitter sigma = {out['jitter_sigma_m']:.4f} m ; lane keep prob = {out['lane_keep_prob']:.3f} -> {PARAMS}")
    journal.event("inject_params", "result", f"jitter sigma {out['jitter_sigma_m']:.4f} m (excess over AV2; nuScenes is SMOOTHER), lane keep prob {out['lane_keep_prob']:.3f}")


# ------------------------------------------------------------------------------------------ injections
def make_transform(cond: str, params: dict, seed: int = 0):
    g = torch.Generator().manual_seed(seed)

    def hz2(inp):
        pos = inp["obj_trajs"]                                          # (B, A, 21, D)
        kf = torch.tensor(KEYFRAMES, device=pos.device)
        xy = pos[..., :2]
        base = xy[:, :, kf, :]                                          # (B, A, 5, 2)
        seg = torch.arange(21, device=pos.device)
        lo = torch.clamp(torch.div(seg, 5, rounding_mode="floor"), max=3)
        w = ((seg - 5 * lo).float() / 5.0).view(1, 1, 21, 1)
        new = base[:, :, lo, :] * (1 - w) + base[:, :, lo + 1, :] * w
        # only agents whose 5 keyframes are all valid are resampled (others keep their observed track)
        m = inp["obj_trajs_mask"]
        ok = m[:, :, kf].all(-1).view(*m.shape[:2], 1, 1)
        pos[..., :2] = torch.where(ok & m.unsqueeze(-1).bool(), new, xy)

    def jitter(inp):
        pos, m = inp["obj_trajs"], inp["obj_trajs_mask"].unsqueeze(-1)
        noise = torch.randn(pos[..., :2].shape, generator=g).to(pos.device) * params["jitter_sigma_m"]
        pos[..., :2] = pos[..., :2] + noise * m

    def map_thin(inp):
        mk = inp["map_polylines_mask"]                                  # (B, 256, 20)
        keep = (torch.rand(mk.shape[:2], generator=g).to(mk.device) < params["lane_keep_prob"])
        inp["map_polylines_mask"] = mk * keep.unsqueeze(-1).to(mk.dtype)

    def both(inp):
        hz2(inp); map_thin(inp)

    return {"hz2": hz2, "jitter": jitter, "map": map_thin, "hz2map": both}[cond]


def cmd_run(a):
    from predict import dump
    from predict_all import best_ckpt, SETS
    params = json.loads(PARAMS.read_text())
    ck = str(best_ckpt(a.model)) if a.ckpt == "auto" else a.ckpt
    db, key = SETS["av2test"]
    for cond in a.conds:
        out = ROOT / "results" / "preds2" / a.model / f"inj_{cond}.npz"
        if out.exists() and not a.force:
            print(f"[skip] {out.name}"); continue
        t0 = time.time()
        journal.event(f"inject_{a.model}_{cond}", "attempt_start", f"ckpt={Path(ck).name} n_batches={a.n_batches}")
        dump(ck, str(ROOT / db), key, f"inj_{cond}", str(out.parent), device=a.device, batch_size=a.batch_size,
             num_workers=a.num_workers, out_file=str(out), limit_batches=a.n_batches,
             transform=make_transform(cond, params))
        journal.event(f"inject_{a.model}_{cond}", "artifact", f"{out.relative_to(ROOT)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sigma"); s.add_argument("--n_batches", type=int, default=80); s.set_defaults(f=cmd_sigma)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True); r.add_argument("--ckpt", default="auto")
    r.add_argument("--conds", nargs="+", default=["hz2", "map", "hz2map"])
    r.add_argument("--n_batches", type=int, default=None); r.add_argument("--device", default="cuda")
    r.add_argument("--batch_size", type=int, default=32); r.add_argument("--num_workers", type=int, default=4)
    r.add_argument("--force", action="store_true"); r.set_defaults(f=cmd_run)
    a = ap.parse_args()
    a.f(a)
