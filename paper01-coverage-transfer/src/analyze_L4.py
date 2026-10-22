"""Addendum L4: multi-dataset training (AB-AV2+NS-s0). Design and success criteria are registered in notes/falsification.md
(Addendum L) BEFORE training finished. Writes results/rev1/L4_multi.json.

Calibration (primary, registered): pooled split CP on the AV2 calibration half plus the nuScenes calibration scenes.
Also reported: worst-domain threshold (max of the per-dataset thresholds) and per-dataset thresholds.
Test: Waymo test (never seen in training or calibration) plus AV2 test and nuScenes test as in-domain controls.
Comparison: AB-AV2-s0 (Waymo gap +23.4) and AB-NS-s0 (+18.3), computed here from the same files.
Success: Waymo gap <= 11.7 points (at most half of AB-AV2-s0's) AND |in-domain gap| <= 3 points on AV2 test and nuScenes test."""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "results" / "preds2"; ALPHA = 0.1
M = "multi_av2ns"


def ld(m, s):
    d = np.load(P / m / f"{s}.npz", allow_pickle=True)
    ids = np.array([(re.search(r"scene-\d+", str(x)) or [str(x)])[0] for x in d["scenario_id"]])
    return d["scores"].astype(float), ids


def qhat(s):
    n = len(s); return float(np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher"))


def boot(flag, ids, reps=1000, seed=0):
    u, inv = np.unique(ids, return_inverse=True)
    s = np.bincount(inv, weights=flag); c = np.bincount(inv)
    rng = np.random.default_rng(seed); g = len(u); v = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, g, g); v[r] = s[i].sum() / c[i].sum()
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def cover(q, scores, ids):
    flag = (scores <= q).astype(float)
    return {"q": q, "coverage": float(flag.mean()), "gap_pts": 100 * (1 - ALPHA - float(flag.mean())), "ci": boot(flag, ids)}


s_av, _ = ld(M, "av2cal"); s_ns, _ = ld(M, "nscal")
thr = {"pooled": qhat(np.concatenate([s_av, s_ns])), "worst_domain": max(qhat(s_av), qhat(s_ns)),
       "av2_only": qhat(s_av), "ns_only": qhat(s_ns)}
tests = {"waymo": ld(M, "wmtest"), "av2_in_domain": ld(M, "av2test"), "ns_in_domain": ld(M, "nstest")}
out = {"model": M, "thresholds_m": thr, "multi": {}}
for cal, q in thr.items():
    out["multi"][cal] = {name: cover(q, s, ids) for name, (s, ids) in tests.items()}

# single-dataset references, same procedure, from the existing predictions
ref = {}
for name, m, cal, wm in (("AB-AV2-s0", "av2_gpu_full", "av2cal", "wm_av2gpu_test"), ("AB-NS-s0", "ns_gpu_full", "nscal", "wm_nsgpu_test")):
    sc, _ = ld(m, cal); sw, iw = ld(m, wm)
    ref[name] = cover(qhat(sc), sw, iw)
out["single_dataset_references_waymo"] = ref

pooled = out["multi"]["pooled"]
a = pooled["waymo"]["gap_pts"]; ai = pooled["av2_in_domain"]["gap_pts"]; ni = pooled["ns_in_domain"]["gap_pts"]
out["criteria"] = {"waymo_gap_pts": a, "waymo_bar_pts": 11.7, "av2_in_domain_gap_pts": ai, "ns_in_domain_gap_pts": ni}
out["success"] = bool(a <= 11.7 and abs(ai) <= 3.0 and abs(ni) <= 3.0)
g = ROOT / "results/rev1/final_minade.json"
(ROOT / "results/rev1/L4_multi.json").write_text(json.dumps(out, indent=1))

print("thresholds (m):", {k: round(v, 2) for k, v in thr.items()})
for cal in ("pooled", "worst_domain"):
    r = out["multi"][cal]
    print(f"{cal:13s} Waymo gap {r['waymo']['gap_pts']:6.1f} (CI {100*(0.9-r['waymo']['ci'][1]):.1f} to {100*(0.9-r['waymo']['ci'][0]):.1f})  "
          f"AV2 {r['av2_in_domain']['gap_pts']:+5.1f}  nuScenes {r['ns_in_domain']['gap_pts']:+5.1f}")
print("references Waymo gap:", {k: round(v["gap_pts"], 1) for k, v in ref.items()})
print("L4 success:", out["success"], f"(Waymo {a:.1f} <= 11.7; |AV2| {abs(ai):.1f} <= 3; |nuScenes| {abs(ni):.1f} <= 3)")
