"""Addendum I outcome: Wayformer second training seed. Writes results/rev1/wf_seed1.json."""
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "results" / "preds2"; ALPHA = 0.1
def qhat(s):
    n = len(s); return float(np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher"))
out = {}
for m in ("av2_wayformer", "av2_wayformer_seed1"):
    q = qhat(np.load(P / m / "av2cal.npz", allow_pickle=True)["scores"].astype(float))
    r = json.loads((ROOT / "results/run2" / f"{m}__forward.json").read_text())["H1_H4"]["0.10"]["raw"]
    wm = np.load(P / m / "wmtest.npz", allow_pickle=True)["scores"].astype(float)
    out[m] = {"q_src": q, "gap_nuscenes": r["target"]["gap"], "gap_nuscenes_ci": r["target"]["ci95"],
              "gap_same": r["same"]["gap"], "gap_waymo": (1 - ALPHA) - float((wm <= q).mean())}
log = (ROOT / "results/validate_wf_seed1_last.log").read_text(errors="ignore")
out["av2_wayformer_seed1"]["val_minADE6_last"] = float(re.search(r"'val/minADE6': ([\d.]+)", log).group(1))
g0, g1 = out["av2_wayformer"]["gap_nuscenes"], out["av2_wayformer_seed1"]["gap_nuscenes"]
out["ratio_nuscenes"] = g1 / g0
out["criterion_met"] = bool(g1 > 0 and 0.5 <= g1 / g0 <= 2.0 and out["av2_wayformer_seed1"]["val_minADE6_last"] <= 0.98)
(ROOT / "results/rev1/wf_seed1.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
