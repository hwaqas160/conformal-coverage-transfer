"""
Addendum B-4 analysis: how much of the real AV2->nuScenes coverage gap does each controlled input shift reproduce?

  G       real gap: AV2-cal threshold on nuScenes (raw score, alpha=.10)
  g_x     gap on the SAME AV2 test scenes after injection x  ( (1-alpha) - coverage under injection )
  g_0     gap on the un-injected AV2 test scenes (same-domain control, ~0)
  effect  g_x - g_0, PAIRED over scenes (bootstrap CI), reported as a fraction of G
  comp    composition-only shift: AV2 test scenes reweighted to the nuScenes distribution of the 7 label-free
          factors (cross-fitted logistic domain-classifier weights, clipped at the 99th percentile)

Registered predictions (notes/falsification.md, Addendum B-4):
  P1 hz2 effect >= 0.25 G (refuted if < 0.10 G for every model)   P2 comp effect < 0.25 G
  P3 hz2+map effect < 0.75 G   (input-level differences do not fully explain the gap)

  python src/analyze_inject.py --models av2_cpu_v1 av2_valsplit_v1 av2_cpu_v2
Output  results/addB/inject_<model>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402
from conformal import split_conformal_quantile  # noqa: E402

PRED = ROOT / "results" / "preds2"
OUT = ROOT / "results" / "addB"
ALPHA = 0.10
CONDS = ("hz2", "map", "hz2map")


def _load(model, name):
    d = np.load(PRED / model / f"{name}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def _align(base: dict, other: dict) -> np.ndarray:
    """Row index into `other` for every row of `base`, by scenario id (loader order differs between runs)."""
    pos = {str(s): i for i, s in enumerate(other["scenario_id"])}
    return np.fromiter((pos[str(s)] for s in base["scenario_id"]), dtype=np.int64, count=len(base["scenario_id"]))


def composition_weights(src_f: np.ndarray, tgt_f: np.ndarray, seed=0) -> np.ndarray:
    """Importance weights p_tgt(x)/p_src(x) for the SOURCE rows from a logistic domain classifier on label-free
    factors, cross-fitted (each source row scored by a classifier that did not see it)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([src_f, tgt_f]); y = np.r_[np.zeros(len(src_f)), np.ones(len(tgt_f))]
    w = np.zeros(len(src_f))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        te_src = te[te < len(src_f)]
        p = clf.predict_proba(sc.transform(X[te_src]))[:, 1]
        w[te_src] = (p / (1 - p)) * (len(src_f) / len(tgt_f))
    return np.minimum(w, np.quantile(w, 0.99))


def run(model: str, B=1000, seed=0) -> dict:
    cal, same, ns = _load(model, "av2cal"), _load(model, "av2test"), _load(model, "ns")
    q = split_conformal_quantile(cal["scores"], ALPHA)
    G = (1 - ALPHA) - float((ns["scores"] <= q).mean())
    cov0 = same["scores"] <= q
    rng = np.random.default_rng(seed)
    out = {"model": model, "alpha": ALPHA, "q": float(q), "G_real_gap": G, "g0_same_domain_gap": (1 - ALPHA) - float(cov0.mean()),
           "injections": {}}
    for c in CONDS:
        p = PRED / model / f"inj_{c}.npz"
        if not p.exists():
            out["injections"][c] = None; continue
        inj = _load(model, f"inj_{c}")
        idx = _align(same, inj)                                    # rows of inj matching `same`
        if len(idx) != len(same["scores"]) or len(set(idx)) != len(idx):
            raise RuntimeError(f"{model}/{c}: injected set does not cover the un-injected test set 1:1")
        d = (cov0.astype(float) - (inj["scores"][idx] <= q).astype(float))   # per-scene loss of coverage due to x
        eff = float(d.mean())
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
        out["injections"][c] = {"n": int(len(d)), "gap_injected": (1 - ALPHA) - float((inj["scores"][idx] <= q).mean()),
                                "effect_pt": eff, "effect_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                "fraction_of_G": eff / G if G else None,
                                "fraction_of_G_ci95": [float(np.percentile(bs, 2.5)) / G, float(np.percentile(bs, 97.5)) / G] if G else None}
    w = composition_weights(same["feats_lf"], ns["feats_lf"])
    covw = float((w * cov0).sum() / w.sum())
    bsw = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(w), len(w)); bsw[b] = (w[i] * cov0[i]).sum() / w[i].sum()
    eff_c = float(cov0.mean() - covw)                              # coverage lost by composition alone
    out["composition"] = {"gap_reweighted": (1 - ALPHA) - covw, "effect_pt": eff_c,
                          "effect_ci95": [float(cov0.mean() - np.percentile(bsw, 97.5)), float(cov0.mean() - np.percentile(bsw, 2.5))],
                          "fraction_of_G": eff_c / G if G else None, "effective_sample_size": float(w.sum() ** 2 / (w ** 2).sum())}
    OUT.mkdir(exist_ok=True)
    (OUT / f"inject_{model}.json").write_text(json.dumps(out, indent=1))
    journal.event(f"inject_analysis_{model}", "result", "; ".join(
        f"{c}: {v['effect_pt']*100:+.1f}pt ({100*v['fraction_of_G']:.0f}% of G={G*100:.1f})" for c, v in
        list(out["injections"].items()) + [("comp", out["composition"])] if v))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); a = ap.parse_args()
    for m in a.models:
        r = run(m)
        print(f"== {m}: real gap G = {100*r['G_real_gap']:+.1f} pt (same-domain {100*r['g0_same_domain_gap']:+.1f})")
        for c, v in list(r["injections"].items()) + [("comp", r["composition"])]:
            if v:
                print(f"   {c:7s} effect {100*v['effect_pt']:+5.1f} pt  CI [{100*v['effect_ci95'][0]:+.1f},{100*v['effect_ci95'][1]:+.1f}]  = {100*v['fraction_of_G']:.0f}% of G")
            else:
                print(f"   {c:7s} (not run)")
