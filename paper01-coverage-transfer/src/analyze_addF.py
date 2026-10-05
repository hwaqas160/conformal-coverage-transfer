"""
Addendum F (notes/falsification.md): descriptive geometric-relevance numbers for a FIXED threshold.
  R(q)  = scene-weighted miss rate (the quantity Proposition 3 controls)            -- reported for reference
  M(q)  = agent-weighted miss share (every agent counts once)                       -- DESCRIPTIVE ONLY, not guaranteed
Radius reported in metres (score units are metres).  Usage: python src/analyze_addF.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
PREDS = ROOT / "results" / "preds2"
ALPHA = 0.10

def load(model, names):
    S, C = [], []
    for n in names:
        d = np.load(PREDS / model / f"{n}.npz", allow_pickle=True)
        S.append(d["scores"].astype(float))
        C.append(np.array([m.group(0) if (m := re.search(r"scene-\d+", str(s))) else str(s) for s in d["scenario_id"]]))
    return np.concatenate(S), np.concatenate(C)

def stats(s, c, q):
    miss = s > q
    u, inv = np.unique(c, return_inverse=True)
    n_j = np.bincount(inv).astype(float)
    per_scene = np.bincount(inv, weights=miss.astype(float)) / n_j
    return {"R_scene_weighted": float(per_scene.mean()), "M_agent_weighted": float(miss.mean()),
            "n_scenes": int(len(u)), "n_agents": int(len(s)),
            "agents_per_scene_median": float(np.median(n_j)), "agents_per_scene_max": int(n_j.max())}

out = {}
for name, model, cal, tgt in [("av2_to_nuscenes", "av2_gpu_full", ["av2cal"], ["ns"]),
                              ("av2_to_waymo", "av2_gpu_full", ["av2cal"], ["wm_av2gpu_test"]),
                              ("ns_to_av2", "ns_gpu_full", ["nscal"], ["av2test"]),
                              ("ns_to_waymo", "ns_gpu_full", ["nscal"], ["wm_nsgpu_test"])]:
    cs, _ = load(model, cal)
    ts, tc = load(model, tgt)
    q = float(np.quantile(cs, 1 - ALPHA))        # standard split-CP threshold on the source calibration split
    r = stats(ts, tc, q); r["q_src_m"] = q; r["radius_m"] = q
    r["R_complement_1_minus_R"] = 1 - r["R_scene_weighted"]
    r["gap_M_vs_1mR"] = r["M_agent_weighted"] - (1 - r["R_scene_weighted"])
    out[name] = r
    print(f"{name:16s} q={q:5.2f}m  R={r['R_scene_weighted']:.3f} (scene-weighted, guaranteed quantity)  "
          f"M={r['M_agent_weighted']:.3f} (agent-weighted, descriptive)  scenes={r['n_scenes']} agents={r['n_agents']} "
          f"agents/scene median={r['agents_per_scene_median']:.0f} max={r['agents_per_scene_max']}")
(ROOT / "results" / "addF").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "addF" / "addF.json").write_text(json.dumps(out, indent=1))
