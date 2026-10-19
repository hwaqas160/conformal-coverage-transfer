"""Addendum K: Method A (worst-domain conformal) and Method B (manoeuvre-conditional calibration).
Design, cells and success criteria are fixed in notes/falsification.md BEFORE this script was run. Writes
results/rev1/methodsAB.json."""
import glob, json, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; P = ROOT / "results" / "preds2"; ALPHA = 0.1
SPEED, HEAD = 4, 5            # feats_lf columns: hist_speed_mean, hist_heading_change


def ld(m, s):
    d = np.load(P / m / f"{s}.npz", allow_pickle=True)
    sid = np.array([(re.search(r"scene-\d+", str(x)) or [str(x)])[0] for x in d["scenario_id"]])
    return d["scores"].astype(float), d["feats_lf"].astype(float), sid


def ldmany(m, names):
    parts = [ld(m, n) for n in names]
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]))


def qhat(s):
    n = len(s)
    return float(np.quantile(s, min(np.ceil((n + 1) * (1 - ALPHA)) / n, 1.0), method="higher"))


def boot(flag, sid, reps=1000, seed=0):
    u, inv = np.unique(sid, return_inverse=True)
    s = np.bincount(inv, weights=flag); c = np.bincount(inv)
    rng = np.random.default_rng(seed); g = len(u); v = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, g, g); v[r] = s[i].sum() / c[i].sum()
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def groups(f, tau):
    slow = f[:, SPEED] < 1.0
    g = np.where(slow, 0, np.where(f[:, HEAD] < tau, 1, 2))
    return g


NS_TRAIN = [Path(x).stem for x in sorted(glob.glob(str(P / "av2_gpu_full" / "nstrain_c*.npz")))]
CELLS = [
    # model, target, single source, extra sources for A  (all labelled calibration data, target excluded)
    ("av2_gpu_full", "ns", "av2cal", ["wm_av2gpu_cal"], "nuScenes"),
    ("av2_gpu_full", "wm_av2gpu_test", "av2cal", NS_TRAIN, "Waymo"),
    ("av2_wayformer", "ns", "av2cal", ["wmcal"], "nuScenes"),
    ("av2_wayformer", "wmtest", "av2cal", NS_TRAIN, "Waymo"),
    ("ns_gpu_full", "av2test", "nscal", ["wm_nsgpu_cal"], "AV2"),
    ("ns_gpu_full", "wm_nsgpu_test", "nscal", ["av2cal"], "Waymo"),
]

out = {"cells": []}
for m, tgt, src, extra, tname in CELLS:
    st, ft, sidt = ld(m, tgt)
    s0, f0, _ = ld(m, src)
    q_src = qhat(s0); q_or = qhat(st)
    row = {"model": m, "target": tname, "n_target": int(len(st)), "q_oracle": q_or, "q_single_source": q_src}

    def ev(q):
        flag = (st <= q).astype(float); cov = float(flag.mean())
        return {"q": q, "coverage": cov, "gap_pts": 100 * (1 - ALPHA - cov), "ci": boot(flag, sidt),
                "area_vs_oracle": (q / q_or) ** 2}

    row["single_source"] = ev(q_src)
    # ---------------- Method A: worst-domain over the other labelled source datasets (+ the training-domain split)
    ex_s = ldmany(m, extra)[0]
    q_ex = qhat(ex_s)
    row["A_worst_domain"] = ev(max(q_src, q_ex)); row["A_worst_domain"]["q_parts"] = {"single_source": q_src, "extra": q_ex}
    row["A_pooled"] = ev(qhat(np.concatenate([s0, ex_s])))
    # ---------------- Method B: Mondrian groups from label-free history features, fixed on the single source
    moving = f0[:, SPEED] >= 1.0
    tau = float(np.median(f0[moving, HEAD]))
    g0 = groups(f0, tau); gt = groups(ft, tau)
    qg = {int(k): qhat(s0[g0 == k]) for k in (0, 1, 2) if (g0 == k).sum() > 20}
    qmap = np.array([qg.get(int(k), q_src) for k in gt])
    flag = (st <= qmap).astype(float); cov = float(flag.mean())
    marg = (st <= q_src).astype(float)
    per = {}
    for k in (0, 1, 2):
        sel = gt == k
        if sel.sum() > 20:
            per[str(k)] = {"n": int(sel.sum()), "q": qg.get(k), "cov_mondrian": float(flag[sel].mean()),
                           "cov_marginal": float(marg[sel].mean())}
    row["B_mondrian"] = {"tau": tau, "coverage": cov, "gap_pts": 100 * (1 - ALPHA - cov), "ci": boot(flag, sidt),
                         "gap_marginal_pts": 100 * (1 - ALPHA - float(marg.mean())), "groups": per,
                         "mean_threshold_ratio_vs_marginal": float(np.mean(qmap / q_src))}
    gm = row["B_mondrian"]["gap_marginal_pts"]; gb = row["B_mondrian"]["gap_pts"]
    row["B_mondrian"]["gap_reduction_fraction"] = (gm - gb) / gm if gm > 0 else float("nan")
    out["cells"].append(row)

cells = out["cells"]
a_ok = [c for c in cells if c["A_worst_domain"]["gap_pts"] <= 2.0 and c["A_worst_domain"]["area_vs_oracle"] <= 2.0]
b_ok = [c for c in cells if c["B_mondrian"]["gap_reduction_fraction"] >= 0.5]
out["A_cells_succeeding"] = len(a_ok); out["B_cells_succeeding"] = len(b_ok)
out["A_success"] = len(a_ok) >= 4; out["B_success"] = len(b_ok) >= 4
(ROOT / "results" / "rev1" / "methodsAB.json").write_text(json.dumps(out, indent=1))

print("cell".ljust(34), "single  worstD(gap,area)   pooled  | B marg->mond (red%)")
for c in cells:
    print(f"{c['model'][:13]:13s}->{c['target']:8s} n={c['n_target']:5d} "
          f"{c['single_source']['gap_pts']:6.1f} {c['A_worst_domain']['gap_pts']:6.1f} ({c['A_worst_domain']['area_vs_oracle']:.2f}x) "
          f"{c['A_pooled']['gap_pts']:6.1f} | {c['B_mondrian']['gap_marginal_pts']:5.1f} -> {c['B_mondrian']['gap_pts']:5.1f} "
          f"({100 * c['B_mondrian']['gap_reduction_fraction']:.0f}%)")
print("A cells succeeding:", len(a_ok), "of 6 ->", "SUCCESS" if out["A_success"] else "FAIL")
print("B cells succeeding:", len(b_ok), "of 6 ->", "SUCCESS" if out["B_success"] else "FAIL")
