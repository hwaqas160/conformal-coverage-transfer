"""Addendum L2: output-based label-free shift monitor. Design and success criteria registered in notes/falsification.md
(Addendum L) BEFORE this script was run. Writes results/rev1/L2_monitor.json."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "results" / "preds2"; ALPHA = 0.1
RNG = np.random.default_rng(0)

# model -> (source calibration file, same-domain test file, out-of-domain target files)
MODELS = {
    "av2_cpu_v1": ("av2cal", "av2test", ["ns"]), "av2_cpu_v2": ("av2cal", "av2test", ["ns"]),
    "av2_gpu_dev_ep11": ("av2cal", "av2test", ["ns"]), "av2_valsplit_v1": ("av2cal", "av2test", ["ns"]),
    "av2_gpu_full": ("av2cal", "av2test", ["ns", "wm_av2gpu_test"]),
    "av2_gpu_full_seed1": ("av2cal", "av2test", ["ns"]),
    "av2_wayformer": ("av2cal", "av2test", ["ns", "wmtest"]),
    "av2_wayformer_seed1": ("av2cal", "av2test", ["ns", "wmtest"]),
    "ns_cpu_v1": ("nscal", "nstest", ["av2test"]),
    "ns_gpu_full": ("nscal", "nstest", ["av2test", "wm_nsgpu_test"]),
}


def ld(m, s):
    d = np.load(P / m / f"{s}.npz", allow_pickle=True)
    pr = d["pred_probs"].astype(float)
    ent = -(pr * np.log(np.clip(pr, 1e-9, 1))).sum(1)
    return d["scores"].astype(float), d["pred_scale"].astype(float).mean(1), ent


def qhat(s):
    n = len(s); return float(np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher"))


rows = []
for m, (src, same, tg) in MODELS.items():
    s0, u0, e0 = ld(m, src); q = qhat(s0)
    for t, kind in [(same, "same")] + [(x, "shift") for x in tg]:
        st, ut, et = ld(m, t)
        gap = 100 * (1 - ALPHA - float((st <= q).mean()))
        rows.append({"model": m, "target": t, "kind": kind, "gap_pts": gap,
                     "S": float(np.log(np.median(ut) / np.median(u0))),
                     "KS": float(ks_2samp(u0, ut).statistic), "dEnt": float(et.mean() - e0.mean())})


def corr(sub, key):
    x = np.array([r[key] for r in sub]); y = np.array([r["gap_pts"] for r in sub])
    rho = float(spearmanr(x, y)[0])
    perm = np.array([spearmanr(x, RNG.permutation(y))[0] for _ in range(10000)])
    return {"n": len(sub), "rho": rho, "p_one_sided": float((perm >= rho).mean())}


out = {"pairs": rows,
       "primary_all_pairs": {k: corr(rows, k) for k in ("S", "KS", "dEnt")},
       "shift_pairs_only": {k: corr([r for r in rows if r["kind"] == "shift"], k) for k in ("S", "KS", "dEnt")}}
pri = out["primary_all_pairs"]["S"]
out["success"] = bool(pri["rho"] >= 0.70 and pri["p_one_sided"] <= 0.05 and pri["rho"] > 0.47)
(ROOT / "results/rev1/L2_monitor.json").write_text(json.dumps(out, indent=1))
for r in rows:
    print(f"{r['model']:20s}{r['target']:15s}{r['kind']:6s} gap={r['gap_pts']:6.1f}  S={r['S']:+.3f}  KS={r['KS']:.3f}  dEnt={r['dEnt']:+.3f}")
for name in ("primary_all_pairs", "shift_pairs_only"):
    print(name, {k: (round(v["rho"], 2), round(v["p_one_sided"], 4), v["n"]) for k, v in out[name].items()})
print("L2 success:", out["success"])
