"""Post-hoc diagnostics for L2 (NOT part of the registered success criterion; labelled as such wherever reported)."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results/rev1/L2_monitor.json").read_text())["pairs"]
sh = [r for r in d if r["kind"] == "shift"]
out = {}
# 1) within a fixed target: does S rank the MODELS by their gap? (removes the between-target effect)
for t in ("ns", "wmtest", "wm_av2gpu_test", "av2test"):
    s = [r for r in sh if r["target"] == t]
    if len(s) >= 4:
        out[f"within_{t}"] = {"n": len(s), "rho_S": float(spearmanr([r["S"] for r in s], [r["gap_pts"] for r in s])[0]),
                              "rho_KS": float(spearmanr([r["KS"] for r in s], [r["gap_pts"] for r in s])[0])}
# 2) detection: flag gap > 3 points (shift pairs) -- AUC of S and KS (rank-based)
def auc(score, y):
    pos = [s for s, v in zip(score, y) if v]; neg = [s for s, v in zip(score, y) if not v]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if pos and neg else float("nan")
y = [r["gap_pts"] > 3 for r in d]
out["auc_flag_gap_gt3_all_pairs"] = {"S": auc([r["S"] for r in d], y), "KS": auc([r["KS"] for r in d], y), "n_pos": int(sum(y)), "n": len(y)}
# 3) leave-one-model-out linear prediction of the gap from S (shift pairs)
models = sorted({r["model"] for r in sh}); err = []
for m in models:
    tr = [r for r in sh if r["model"] != m]; te = [r for r in sh if r["model"] == m]
    b, a = np.polyfit([r["S"] for r in tr], [r["gap_pts"] for r in tr], 1)
    err += [abs(a + b * r["S"] - r["gap_pts"]) for r in te]
out["loo_model_linear_MAE_pts"] = float(np.mean(err)); out["gap_range_pts"] = [min(r["gap_pts"] for r in sh), max(r["gap_pts"] for r in sh)]
(ROOT / "results/rev1/L2_diagnostics.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
