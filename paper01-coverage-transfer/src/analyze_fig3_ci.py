"""Scene-cluster-aware 95% intervals for Fig. 3 (coverage by manoeuvre, AB-AV2-s0: AV2 test vs nuScenes Boston vs Singapore).

The earlier figure used agent-level Wilson intervals, which ignore within-scene dependence on nuScenes (about 65 agents per scene).
Here every interval is a scene-cluster bootstrap (1000 resamples of whole scenes within the city), and the AV2 test split uses the
same procedure with one cluster per independent clip. Writes results/rev1/fig3_cluster_ci.json."""
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import analyze_addB as B  # noqa: E402
from conformal import split_conformal_quantile  # noqa: E402

ROOT = SRC.parent
ALPHA = 0.10
cal, same, tgt, _ = B.sets_for("av2_gpu_full", "forward")
q = split_conformal_quantile(cal["scores"], ALPHA)
loc = json.loads((B.OUT / "nuscenes_scene_location.json").read_text())
scene_t = B.clusters_of(tgt); scene_s = B.clusters_of(same)
city = np.array(["singapore" if loc[s].startswith("singapore") else "boston" for s in scene_t])


def cluster_ci(flag, ids, reps=1000, seed=0):
    u, inv = np.unique(ids, return_inverse=True)
    s = np.bincount(inv, weights=flag); c = np.bincount(inv)
    rng = np.random.default_rng(seed); g = len(u); v = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, g, g)
        v[r] = s[i].sum() / max(1, c[i].sum())
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


out = {"q_src": float(q), "by_type": {}}
for ti, nm in enumerate(B.TRAJ_TYPES):
    sel_s = same["trajectory_type"] == ti
    if sel_s.sum() < 40:
        continue
    row = {}
    fs = (same["scores"][sel_s] <= q).astype(float)
    row["same"] = {"cov": float(fs.mean()), "n": int(sel_s.sum()), "ci": cluster_ci(fs, scene_s[sel_s])}
    for c in ("boston", "singapore"):
        sel = (tgt["trajectory_type"] == ti) & (city == c)
        if sel.sum() < 30:
            row[c] = None; continue
        f = (tgt["scores"][sel] <= q).astype(float)
        row[c] = {"cov": float(f.mean()), "n": int(sel.sum()), "n_scenes": int(len(np.unique(scene_t[sel]))),
                  "ci": cluster_ci(f, scene_t[sel])}
    out["by_type"][nm] = row
(ROOT / "results" / "rev1" / "fig3_cluster_ci.json").write_text(json.dumps(out, indent=1))
for nm, r in out["by_type"].items():
    print(f"{nm:15s}", *(f"{k}: {v['cov']:.2f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}] n={v['n']}" if v else f"{k}: --" for k, v in r.items()))
