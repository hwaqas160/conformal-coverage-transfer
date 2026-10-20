"""Addendum L1: sequential (anytime-valid) scene-level certification. Design and success criteria are registered in
notes/falsification.md (Addendum L) BEFORE this script was run. Writes results/rev1/L1_sequential.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import analyze_rev1 as A  # noqa: E402

ALPHA, DELTA = A.ALPHA, A.DELTA
T_MIN, STABLE_LAG, STABLE_TOL, T_CAP = 30, 10, 0.03, 300
DRAWS = 1000


def scene_loss_matrix(groups, Q):
    """(n_scenes, len(Q)) per-scene miscoverage L_j(q) for every candidate threshold."""
    return np.stack([1.0 - np.searchsorted(g, Q, side="right") / len(g) for g in groups])


def sequential_run(Lmat, order, grid, q_src):
    """Run the anytime-valid betting LTT scene by scene; return (stop_t, q_at_stop, capped)."""
    Q = np.concatenate([[q_src], grid])
    mu = np.full(Q.shape, ALPHA / 2); var = np.full(Q.shape, ALPHA / 4); n = 1.0
    k = np.ones(Q.shape); kmax = np.ones(Q.shape)
    hist = []
    for t, j in enumerate(order[:T_CAP], start=1):
        x = Lmat[j]
        edge = ALPHA - mu
        lam = np.where(edge <= 0, 0.0, np.minimum(edge / (var + edge * edge), A.BET_CAP / (1.0 - ALPHA)))
        k = k * (1.0 + lam * (ALPHA - x))
        assert (k >= 0).all()
        kmax = np.maximum(kmax, k)
        n += 1; d = x - mu; mu = mu + d / n; var = var + (d * (x - mu) - var) / n
        p = np.minimum(1.0, 1.0 / kmax)             # running-maximum p-value: valid at any stopping time (Ville)
        if p[0] <= DELTA / 2:
            q_t = float(q_src)
        else:
            rej = p[1:] <= DELTA / 2
            if not rej[0]:
                q_t = float("inf")
            else:
                ff = int(np.argmin(rej)) if not rej.all() else len(rej)
                q_t = float(grid[ff - 1])
        hist.append(q_t)
        if t >= T_MIN and np.isfinite(q_t) and t - STABLE_LAG >= 1:
            prev = hist[t - STABLE_LAG - 1]
            if np.isfinite(prev) and abs(q_t - prev) <= STABLE_TOL * prev:
                return t, q_t, False
    return min(len(order), T_CAP), hist[-1], True


def fixed_budget(pair):
    sc = json.loads((A.ROOT / "results/rev1" / f"scene_{pair}.json").read_text())["by_m"]
    for m in sorted(sc, key=int):
        b = sc[m]["bet"]
        if b["median_area_vs_oracle"] is not None and b["median_area_vs_oracle"] <= 2.0 and b["frac_infinite"] <= 0.05:
            return int(m)
    return None


def run(pair, draws=DRAWS, seed=0):
    model, cal_name, lab_names, ev_name = A.PAIRS[pair]
    t0 = time.time()
    cal_s, _ = A.load_scores(model, [cal_name])
    lab_s, lab_c = A.load_scores(model, lab_names)
    ev_s, ev_c = A.load_scores(model, [ev_name])
    lab, ev = A.Pool(lab_s, lab_c), A.Pool(ev_s, ev_c)
    q_src = float(A.split_conformal_quantile(cal_s, ALPHA))
    grid = q_src * np.exp(np.linspace(np.log(4.0), np.log(0.5), 300))
    Lmat = scene_loss_matrix(lab.groups, np.concatenate([[q_src], grid]))
    rng = np.random.default_rng(seed)
    q_or = ev.oracle(ALPHA)
    stops, areas, ok, inf, capped = [], [], 0, 0, 0
    for _ in range(draws):
        order = rng.permutation(len(lab.scene_ids))
        t, q, cap = sequential_run(Lmat, order, grid, q_src)
        stops.append(t); capped += cap
        ok += ev.R(q) <= ALPHA
        if np.isfinite(q):
            areas.append((q / q_or) ** 2)
        else:
            inf += 1
    fb = fixed_budget(pair)
    res = {"pair": pair, "draws": draws, "p_valid": ok / draws, "frac_infinite_at_stop": inf / draws,
           "frac_capped": capped / draws, "median_stop": float(np.median(stops)),
           "q25_stop": float(np.percentile(stops, 25)), "q75_stop": float(np.percentile(stops, 75)),
           "q90_stop": float(np.percentile(stops, 90)),
           "median_area_vs_oracle": float(np.median(areas)) if areas else None, "fixed_budget_m": fb,
           "stop_over_fixed": (float(np.median(stops)) / fb) if fb else None}
    print(f"{pair:22s} valid={res['p_valid']:.3f} inf={res['frac_infinite_at_stop']:.2f} capped={res['frac_capped']:.2f} "
          f"median stop={res['median_stop']:.0f} (IQR {res['q25_stop']:.0f}-{res['q75_stop']:.0f}) fixed m={fb} "
          f"area={res['median_area_vs_oracle']} ({time.time()-t0:.0f}s)", flush=True)
    return res


if __name__ == "__main__":
    out = {"design": {"t_min": T_MIN, "stable_lag": STABLE_LAG, "stable_tol": STABLE_TOL, "t_cap": T_CAP, "alpha": ALPHA, "delta": DELTA},
           "pairs": {}}
    for pair in A.PAIRS:
        out["pairs"][pair] = run(pair)
    ok = [p for p in out["pairs"].values()
          if p["p_valid"] >= 1 - DELTA and p["median_area_vs_oracle"] is not None and p["median_area_vs_oracle"] <= 2.0
          and p["stop_over_fixed"] is not None and p["stop_over_fixed"] <= 0.75]
    out["transfers_succeeding"] = len(ok); out["success"] = len(ok) >= 4
    (A.ROOT / "results/rev1/L1_sequential.json").write_text(json.dumps(out, indent=1))
    print("L1 transfers succeeding:", len(ok), "of 6 ->", "SUCCESS" if out["success"] else "FAIL")
