"""Addendum L3: robust conformal over a half-chi-square ball (Cauchois et al. 2024) as a baseline, radius estimated
leave-one-dataset-out. Design and success criteria registered in notes/falsification.md (Addendum L) BEFORE this script was run.
Writes results/rev1/L3_robust.json.

Closed form: for f(t) = (t-1)^2/2 and an event with source probability p, the worst-case probability over {Q: D_f(Q||P) <= rho} is
q_min = p - sqrt(2 rho p (1-p)).  The robust threshold is the empirical quantile at the smallest p with q_min >= 1 - alpha."""
import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "results" / "preds2"; ALPHA = 0.1; NBINS = 20


def sc(m, names):
    return np.concatenate([np.load(P / m / f"{n}.npz", allow_pickle=True)["scores"].astype(float) for n in names])


def sid(m, n):
    d = np.load(P / m / f"{n}.npz", allow_pickle=True)
    return np.array([(re.search(r"scene-\d+", str(x)) or [str(x)])[0] for x in d["scenario_id"]])


def qhat_level(s, level):
    n = len(s); return float(np.quantile(s, min(np.ceil((n + 1) * level) / n, 1.0), method="higher"))


def p_star(rho):
    """smallest p in [1-alpha, 1] with p - sqrt(2 rho p (1-p)) >= 1-alpha (bisection; p=1 always qualifies)."""
    lo, hi = 1 - ALPHA, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if mid - np.sqrt(2 * rho * mid * (1 - mid)) >= 1 - ALPHA:
            hi = mid
        else:
            lo = mid
    return hi


def rho_for_p(p):
    """the radius at which the robust threshold sits exactly at source-CDF value p (inverse of p_star)."""
    return (p - (1 - ALPHA)) ** 2 / (2 * p * (1 - p)) if p < 1 else float("inf")


def binned_halfchi2(src, other):
    edges = np.quantile(src, np.linspace(0, 1, NBINS + 1)[1:-1])
    q = np.bincount(np.searchsorted(edges, other, side="right"), minlength=NBINS) / len(other)
    p = np.full(NBINS, 1.0 / NBINS)
    return float(0.5 * np.sum((q - p) ** 2 / p))


def boot(flag, ids, reps=1000, seed=0):
    u, inv = np.unique(ids, return_inverse=True)
    s = np.bincount(inv, weights=flag); c = np.bincount(inv)
    rng = np.random.default_rng(seed); g = len(u); v = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, g, g); v[r] = s[i].sum() / c[i].sum()
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


NS_TRAIN = [Path(x).stem for x in sorted(glob.glob(str(P / "av2_gpu_full" / "nstrain_c*.npz")))]
CELLS = [("av2_gpu_full", "ns", "av2cal", ["wm_av2gpu_cal"], "nuScenes"),
         ("av2_gpu_full", "wm_av2gpu_test", "av2cal", NS_TRAIN, "Waymo"),
         ("av2_wayformer", "ns", "av2cal", ["wmcal"], "nuScenes"),
         ("av2_wayformer", "wmtest", "av2cal", NS_TRAIN, "Waymo"),
         ("ns_gpu_full", "av2test", "nscal", ["wm_nsgpu_cal"], "AV2"),
         ("ns_gpu_full", "wm_nsgpu_test", "nscal", ["av2cal"], "Waymo")]

out = {"cells": []}
print("cell".ljust(28), "rho_hat  rho*     single  robust(gap,area)")
for m, tgt, src, extra, tname in CELLS:
    s0 = sc(m, [src]); st = sc(m, [tgt]); ids = sid(m, tgt)
    q_or = qhat_level(st, 1 - ALPHA); q_single = qhat_level(s0, 1 - ALPHA)
    rho_hat = binned_halfchi2(s0, sc(m, extra))
    p_or = float((s0 <= q_or).mean()); rho_star = rho_for_p(p_or)

    def ev(q):
        flag = (st <= q).astype(float)
        return {"q": q, "coverage": float(flag.mean()), "gap_pts": 100 * (1 - ALPHA - float(flag.mean())),
                "ci": boot(flag, ids), "area_vs_oracle": (q / q_or) ** 2}

    q_rob = qhat_level(s0, p_star(rho_hat))
    row = {"model": m, "target": tname, "rho_hat_LODO": rho_hat, "rho_oracle": rho_star, "p_star": p_star(rho_hat),
           "single_source": ev(q_single), "robust": ev(q_rob)}
    out["cells"].append(row)
    print(f"{m[:13]:13s}->{tname:9s}    {rho_hat:7.3f}  {rho_star:7.3f}  {row['single_source']['gap_pts']:6.1f}  "
          f"{row['robust']['gap_pts']:6.1f} ({row['robust']['area_vs_oracle']:.2f}x)")
ok = [c for c in out["cells"] if c["robust"]["gap_pts"] <= 2.0 and c["robust"]["area_vs_oracle"] <= 2.0]
out["cells_succeeding"] = len(ok); out["success"] = len(ok) >= 4
(ROOT / "results/rev1/L3_robust.json").write_text(json.dumps(out, indent=1))
print("L3 cells succeeding:", len(ok), "of 6 ->", "SUCCESS" if out["success"] else "FAIL")
