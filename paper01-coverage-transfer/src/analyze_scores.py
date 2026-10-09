"""Score ablation (REVISION_PLAN phase 1, item 4): is the transfer gap specific to the registered score?
Each alternative score is calibrated on the source cal split and evaluated on target test sets at alpha=0.1.
Exploratory; the registered score remains the primary one. Time weights for S3 are fitted on a disjoint half of the
source calibration set so the conformal quantile is computed on data independent of the weights."""
import json, sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "results" / "preds2"
ALPHA = 0.1
SETS = {  # model -> (cal, [targets])
    "av2_gpu_full": ("av2cal", ["av2test", "ns", "wm_av2gpu_test"]),
    "av2_wayformer": ("av2cal", ["av2test", "ns", "wmtest"]),
    "ns_gpu_full": ("nscal", ["nstest", "av2test", "wm_nsgpu_test"]),
}
GAMMA = 0.5

def load(m, s):
    d = np.load(P / m / f"{s}.npz", allow_pickle=True)
    return {k: d[k] for k in ("pred_trajs", "pred_probs", "gt", "gt_mask", "scenario_id")}

def errs(d):
    e = np.linalg.norm(d["pred_trajs"] - d["gt"][:, None], axis=-1)     # N,K,T
    return e

def scores(d, w=None):
    e = errs(d); msk = d["gt_mask"][:, None, :]; N, K, T = e.shape
    out = {}
    worst = np.where(msk, e, -np.inf).max(-1)                            # N,K
    out["S0_registered"] = worst.min(1)
    top = d["pred_probs"].argmax(1)
    out["S1_top_prob_mode"] = worst[np.arange(N), top]
    last = d["gt_mask"].shape[1] - 1 - np.argmax(d["gt_mask"][:, ::-1], axis=1)
    out["S2_final_displacement"] = e[np.arange(N)[:, None], np.arange(K)[None], last[:, None]].min(1)
    out["S4_prob_weighted"] = (worst / np.maximum(d["pred_probs"], 1e-3) ** GAMMA).min(1)
    if w is not None:
        out["S3_time_scaled"] = np.where(msk, e / w[None, None, :], -np.inf).max(-1).min(1)
    return out

def time_weights(d):
    e = errs(d); best = e[np.arange(len(e)), np.where(d["gt_mask"][:, None, :], e, np.inf).max(-1).argmin(1)]
    w = np.array([np.median(best[d["gt_mask"][:, t], t]) for t in range(best.shape[1])])
    return np.maximum(w, 1e-2)

def qhat(s):
    n = len(s); lvl = min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0)
    return float(np.quantile(s, lvl, method="higher"))

def boot_ci(cov_flags, sid, reps=500, seed=0):
    u, inv = np.unique(sid, return_inverse=True)
    sums = np.bincount(inv, weights=cov_flags); cnt = np.bincount(inv)
    rng = np.random.default_rng(seed); g = len(u); vals = []
    for _ in range(reps):
        i = rng.integers(0, g, g); vals.append(sums[i].sum() / cnt[i].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def main():
    res = {}
    for m, (cal, tgts) in SETS.items():
        c = load(m, cal); rng = np.random.default_rng(0)
        perm = rng.permutation(len(c["gt"])); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
        sub = lambda d, ix: {k: v[ix] for k, v in d.items()}
        w = time_weights(sub(c, A)); sc_cal = scores(sub(c, B), w)
        w_mean = float(w[np.isfinite(w)].mean())
        res[m] = {"n_cal": int(len(B)), "mean_time_weight_m": w_mean}
        for name, s in sc_cal.items():
            q = qhat(s); row = {"q": q}
            if name == "S3_time_scaled": row["mean_tube_radius_m"] = q * w_mean
            for t in tgts:
                d = load(m, t); st = scores(d, w)[name]; flag = (st <= q).astype(float)
                cov = float(flag.mean())
                row[t] = {"coverage": cov, "gap_pts": 100 * ((1 - ALPHA) - cov), "ci": boot_ci(flag, d["scenario_id"]), "n": int(len(st))}
            res[m][name] = row
    (ROOT / "results" / "rev1").mkdir(exist_ok=True)
    (ROOT / "results" / "rev1" / "score_ablation.json").write_text(json.dumps(res, indent=1))
    for m, r in res.items():
        print(m)
        for name, row in r.items():
            if not isinstance(row, dict) or "q" not in row: continue
            print(f"  {name:24s} q={row['q']:7.2f} " + " ".join(f"{t}:{row[t]['coverage']:.3f}" for t in SETS[m][1]))
main()
