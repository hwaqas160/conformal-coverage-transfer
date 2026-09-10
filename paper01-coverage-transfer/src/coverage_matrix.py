"""
Paper 01 core experiment: the calibrate-on-A / deploy-on-B coverage transfer matrix,
plus the H2 attribution regression and H3 recalibration comparison.

Consumes the .npz files written by predict.py (one per (model_source, eval_dataset) pair).
By convention a file tagged  "<evalkey>_from_<srckey>"  holds predictions of the model
trained on <srckey>, evaluated on <evalkey>.

Outputs JSON + CSV tables into results/.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

import sys
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from conformal import (  # noqa: E402
    SplitConformal, WeightedSplitConformal, NormalizedSplitConformal,
    GroupConditionalConformal, domain_classifier_weights,
)


def _load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _split_idx(n, frac_cal=0.5, seed=0):
    """within an in-domain eval set, split rows into calibration / test halves."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(n * frac_cal)
    return perm[:k], perm[k:]


def coverage_transfer(preds_dir, alphas=(0.05, 0.10, 0.20), seeds=(0, 1, 2),
                      out_dir="results"):
    """
    preds_dir holds files like  av2_from_av2.npz, ns_from_av2.npz, av2_from_ns.npz, ...
    For each SOURCE model we calibrate SCP on the in-domain (source==eval) file's
    calibration half, then measure coverage on:
      * source in-domain test half   -> Delta_in
      * every other dataset's file    -> Delta_cross
    """
    preds_dir = Path(preds_dir)
    files = {p.stem: p for p in preds_dir.glob("*.npz")}
    # discover (eval, src) pairs
    pairs = {}
    for stem in files:
        if "_from_" not in stem:
            continue
        ev, src = stem.split("_from_")
        pairs[(ev, src)] = files[stem]
    sources = sorted({s for _, s in pairs})
    evals = sorted({e for e, _ in pairs})
    print(f"[coverage] sources={sources} evals={evals}")

    rows = []
    for src in sources:
        if (src, src) not in pairs:
            print(f"  !! no in-domain file for source {src}; skipping its calibration")
            continue
        indom = _load(pairs[(src, src)])
        for alpha in alphas:
            for seed in seeds:
                cal_i, test_i = _split_idx(len(indom["scores"]), 0.5, seed)
                sc = SplitConformal(alpha=alpha).calibrate(indom["scores"][cal_i])
                # in-domain
                r_in = sc.evaluate(indom["scores"][test_i])
                rows.append(dict(source=src, eval=src, kind="in", alpha=alpha, seed=seed,
                                 nominal=1 - alpha, **r_in,
                                 delta=(1 - alpha) - r_in["coverage"]))
                # cross-domain
                for ev in evals:
                    if ev == src or (ev, src) not in pairs:
                        continue
                    cross = _load(pairs[(ev, src)])
                    r_x = sc.evaluate(cross["scores"])
                    rows.append(dict(source=src, eval=ev, kind="cross", alpha=alpha,
                                     seed=seed, nominal=1 - alpha, **r_x,
                                     delta=(1 - alpha) - r_x["coverage"]))

    _write_tables(rows, out_dir, "coverage_transfer")
    return rows


def recalibration_compare(preds_dir, alpha=0.10, seeds=(0, 1, 2), out_dir="results"):
    """H3: for every cross pair, compare uncorrected SCP vs weighted / normalized / group."""
    preds_dir = Path(preds_dir)
    files = {p.stem: p for p in preds_dir.glob("*.npz")}
    pairs = {}
    for stem in files:
        if "_from_" in stem:
            ev, src = stem.split("_from_")
            pairs[(ev, src)] = files[stem]
    sources = sorted({s for _, s in pairs})

    rows = []
    for src in sources:
        if (src, src) not in pairs:
            continue
        indom = _load(pairs[(src, src)])
        for ev, s in list(pairs):
            if s != src or ev == src:
                continue
            tgt = _load(pairs[(ev, src)])
            for seed in seeds:
                cal_i, _ = _split_idx(len(indom["scores"]), 0.5, seed)
                cs, cf = indom["scores"][cal_i], indom["feats"][cal_i]
                ts, tf = tgt["scores"], tgt["feats"]

                base = SplitConformal(alpha).calibrate(cs).evaluate(ts)

                w = domain_classifier_weights(cf, tf, seed=seed)
                wt = WeightedSplitConformal(alpha).calibrate(cs, w).evaluate(ts)

                nm = NormalizedSplitConformal(alpha).calibrate(cs, cf).evaluate(ts, tf)

                # group by agent-density tertile (factor index 1)
                q1, q2 = np.percentile(cf[:, 1], [33, 66])
                cg = np.digitize(cf[:, 1], [q1, q2])
                tg = np.digitize(tf[:, 1], [q1, q2])
                gr = GroupConditionalConformal(alpha).calibrate(cs, cg).evaluate(ts, tg)

                for method, res in [("uncorrected", base), ("weighted", wt),
                                    ("normalized", nm), ("group", gr)]:
                    rows.append(dict(source=src, eval=ev, seed=seed, method=method,
                                     alpha=alpha, nominal=1 - alpha,
                                     coverage=res["coverage"], q_hat=res["q_hat"],
                                     delta=(1 - alpha) - res["coverage"]))
    _write_tables(rows, out_dir, "recalibration")
    return rows


def _write_tables(rows, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(rows, indent=2))
    if rows:
        cols = list(rows[0].keys())
        lines = [",".join(cols)]
        for r in rows:
            lines.append(",".join(str(r.get(c, "")) for c in cols))
        (out_dir / f"{name}.csv").write_text("\n".join(lines))
    # console summary
    if name == "coverage_transfer" and rows:
        import statistics as st
        for kind in ("in", "cross"):
            ds = [r["delta"] for r in rows if r["kind"] == kind and abs(r["alpha"] - 0.1) < 1e-6]
            if ds:
                print(f"  [{kind}] alpha=0.10  mean delta = {st.mean(ds):+.4f}  "
                      f"(min {min(ds):+.3f}, max {max(ds):+.3f}, n={len(ds)})")
    print(f"  wrote {out_dir/f'{name}.json'} and .csv ({len(rows)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(SRC.parent / "results" / "preds"))
    ap.add_argument("--out", default=str(SRC.parent / "results"))
    ap.add_argument("--mode", choices=["coverage", "recal", "both"], default="both")
    a = ap.parse_args()
    if a.mode in ("coverage", "both"):
        coverage_transfer(a.preds, out_dir=a.out)
    if a.mode in ("recal", "both"):
        recalibration_compare(a.preds, out_dir=a.out)
