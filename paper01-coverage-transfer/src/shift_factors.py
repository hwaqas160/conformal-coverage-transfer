"""
The 7 pre-registered shift factors (notes/falsification.md, H2), computed per predicted
agent from a UniTraj batch dict.

All factors are cheap functions of the model INPUT + ground truth geometry -- they do NOT
use the model prediction, so they can be computed once per dataset and reused.

UniTraj input layout (post-collate, cfg defaults, VEHICLE agents):
  obj_trajs        (B, A, 21, C)   agent history, ego-centric & rotated; C~=39
                                   [...,:2]  = (x, y)
                                   [...,25:27] = (vx, vy)   (velocity NOT masked by default)
                                   [...,35:37] = one-hot-ish time encoding tail (varies)
  obj_trajs_mask   (B, A, 21) bool
  track_index_to_predict (B,)      index of ego agent in A
  center_gt_trajs  (B, 60, 4)      future in ego frame [x, y, vx, vy]
  center_gt_trajs_mask (B, 60) bool
  map_polylines    (B, 256, 20, 29)
  map_polylines_mask (B, 256, 20) bool
  kalman_difficulty (B, ...)       precomputed
  trajectory_type  (B,)            precomputed class id (stationary/straight/turns)

dt handling: after ScenarioNet + UniTraj resampling every dataset is at 0.1 s steps, so
factor 1 uses the ORIGINAL native dt passed in explicitly (AV2=0.1, nuScenes=0.5, Waymo=0.1,
nuPlan=0.05) -- recorded per dataset in DATASET_DT.
"""
from __future__ import annotations

import numpy as np

DATASET_DT = {  # native sampling period (s) before resampling
    "av2": 0.1, "av2_train": 0.1, "av2_val": 0.1,
    "nuscenes": 0.5, "ns": 0.5,
    "waymo": 0.1, "wo": 0.1,
    "nuplan": 0.05, "np": 0.05,
}

FACTOR_NAMES = [
    "native_dt",
    "n_agents_near_ego",
    "map_point_density",
    "n_lanes_near_ego",
    "ego_speed_mean",
    "ego_turning_frac_proxy",
    "gt_future_curvature",
]


def _ego_rows(batch_np):
    """helper: return (B,) index array for ego + convenience views as numpy."""
    inp = batch_np
    return inp


def compute_factors(batch_np: dict, dataset_key: str, near_radius: float = 50.0) -> np.ndarray:
    """
    batch_np: dict of numpy arrays for ONE batch (already .cpu().numpy()).
    Returns (B, 7) float array, columns = FACTOR_NAMES.
    """
    obj = batch_np["obj_trajs"]              # (B, A, 21, C)
    objm = batch_np["obj_trajs_mask"].astype(bool)   # (B, A, 21)
    gt = batch_np["center_gt_trajs"]         # (B, 60, 4)
    gtm = batch_np["center_gt_trajs_mask"].astype(bool)  # (B, 60)
    mp = batch_np["map_polylines"]           # (B, 256, 20, 29)
    mpm = batch_np["map_polylines_mask"].astype(bool)    # (B, 256, 20)

    B, A, H, C = obj.shape
    last = H - 1

    # --- factor 1: native dt (constant per dataset) ---
    dt = DATASET_DT.get(str(dataset_key).lower(), 0.1)
    f_dt = np.full(B, dt, dtype=np.float32)

    # --- factor 2: # agents present near ego at last observed step (within near_radius) ---
    # ego is at origin in the ego-centric frame; use each agent's last-observed xy
    xy_last = obj[:, :, last, :2]                    # (B, A, 2)
    present = objm[:, :, last]                       # (B, A)
    dist = np.linalg.norm(xy_last, axis=-1)          # (B, A)
    near = present & (dist <= near_radius)
    f_nagents = near.sum(axis=1).astype(np.float32) - 1.0  # exclude ego
    f_nagents = np.clip(f_nagents, 0, None)

    # --- factor 3: map point density (valid polyline points per 100 m^2 in the crop) ---
    valid_pts = mpm.sum(axis=(1, 2)).astype(np.float32)     # (B,)
    # crop area from the spread of valid map points
    area = np.full(B, 1.0, np.float32)
    for i in range(B):
        pts = mp[i][mpm[i]][:, :2]
        if len(pts) >= 3:
            span = (pts.max(0) - pts.min(0))
            area[i] = max(span[0] * span[1], 1.0)
    f_mapdens = valid_pts / (area / 100.0)

    # --- factor 4: # distinct lane polylines with any valid point near ego (<= near_radius) ---
    lane_has_near = np.zeros(B, np.float32)
    for i in range(B):
        m = mpm[i]                                   # (256, 20)
        if not m.any():
            continue
        pl_xy = mp[i][..., :2]                        # (256, 20, 2)
        d = np.linalg.norm(pl_xy, axis=-1)           # (256, 20)
        near_pl = (m & (d <= near_radius)).any(axis=1)
        lane_has_near[i] = near_pl.sum()
    f_nlanes = lane_has_near

    # --- factor 5: ego speed (mean over GT future, from [vx, vy]) ---
    v = gt[:, :, 2:4]                                 # (B, 60, 2)
    spd = np.linalg.norm(v, axis=-1)                 # (B, 60)
    spd = np.where(gtm, spd, np.nan)
    f_speed = np.nanmean(spd, axis=1).astype(np.float32)
    f_speed = np.nan_to_num(f_speed, nan=0.0)

    # --- factor 6: turning proxy = |net heading change| over GT future > 30 deg ? ---
    # heading from displacement of GT positions
    p = gt[:, :, :2]
    turn = np.zeros(B, np.float32)
    for i in range(B):
        valid = np.where(gtm[i])[0]
        if len(valid) < 5:
            continue
        pp = p[i, valid]
        d = np.diff(pp, axis=0)
        d = d[np.linalg.norm(d, axis=1) > 1e-3]
        if len(d) < 2:
            continue
        ang = np.arctan2(d[:, 1], d[:, 0])
        net = np.abs(np.angle(np.exp(1j * (ang[-1] - ang[0]))))
        turn[i] = float(net > np.deg2rad(30))
    f_turn = turn

    # --- factor 7: mean absolute curvature of GT future (rad per metre) ---
    curv = np.zeros(B, np.float32)
    for i in range(B):
        valid = np.where(gtm[i])[0]
        if len(valid) < 5:
            continue
        pp = p[i, valid]
        d = np.diff(pp, axis=0)
        seglen = np.linalg.norm(d, axis=1)
        keep = seglen > 1e-3
        d, seglen = d[keep], seglen[keep]
        if len(d) < 2:
            continue
        ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        dtheta = np.abs(np.diff(ang))
        s = 0.5 * (seglen[:-1] + seglen[1:])
        curv[i] = float(np.sum(dtheta) / max(np.sum(s), 1e-3))
    f_curv = curv

    return np.stack([f_dt, f_nagents, f_mapdens, f_nlanes, f_speed, f_turn, f_curv], axis=1).astype(np.float32)


def summarize(feats: np.ndarray) -> dict:
    """per-dataset factor summary for the H2 table."""
    out = {}
    for j, name in enumerate(FACTOR_NAMES):
        col = feats[:, j]
        out[name] = {
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "p10": float(np.percentile(col, 10)),
            "p50": float(np.percentile(col, 50)),
            "p90": float(np.percentile(col, 90)),
        }
    return out


if __name__ == "__main__":
    # exercised via predict.py on real batches; here just a shape check on fakes
    B = 8
    fake = {
        "obj_trajs": np.random.randn(B, 15, 21, 39).astype(np.float32),
        "obj_trajs_mask": (np.random.rand(B, 15, 21) > 0.3),
        "center_gt_trajs": np.cumsum(np.random.randn(B, 60, 4) * 0.3, axis=1).astype(np.float32),
        "center_gt_trajs_mask": np.ones((B, 60), bool),
        "map_polylines": np.random.randn(B, 256, 20, 29).astype(np.float32) * 10,
        "map_polylines_mask": (np.random.rand(B, 256, 20) > 0.5),
    }
    f = compute_factors(fake, "av2")
    print("factors shape:", f.shape)
    for n, v in zip(FACTOR_NAMES, f.mean(0)):
        print(f"  {n:24s} {v:.3f}")
