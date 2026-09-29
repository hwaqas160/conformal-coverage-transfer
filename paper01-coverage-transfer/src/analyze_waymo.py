"""
Addendum C (notes/falsification.md, registered 2026-09-28 BEFORE any Waymo prediction existed): H17-H20 on the
third dataset (Waymo Open Motion v1.2.1 validation, 30/150 shards, scenario-level cal/test split, salt "wm1").

  H17  does the guarantee break on a third dataset: source-calibrated coverage on Waymo test, scenario-cluster CI
  H18  same source model, gap > 0 on BOTH targets (nuScenes and Waymo)
  H19  does the scene-level fix (Prop. 3, betting LTT) transfer: m in {30,40,50,60} labelled Waymo scenarios
  H20  does the model-normalised score reduce (not remove) the gap on Waymo, as it does on nuScenes

Usage:  python src/analyze_waymo.py --model av2_gpu_full --cal av2cal
        python src/analyze_waymo.py --model ns_gpu_full  --cal nscal
Output: results/addC/<model>.json
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
from analyze_run2 import load, boot_gap  # noqa: E402
from analyze_addB import clusters_of, h16_scene_ltt  # noqa: E402
from conformal import split_conformal_quantile  # noqa: E402
from solutions import normalized_scores  # noqa: E402

ALPHA, DELTA = 0.10, 0.10
OUT = ROOT / "results" / "addC"


def boot_gap_clustered(cal_s, tgt_s, tgt_clusters, alpha, B=1000, seed=0):
    """Same as analyze_run2.boot_gap but the TARGET resample is by scenario (Waymo scenarios carry several
    correlated agents each, like nuScenes scenes) -- required because H17/H18 are cluster-level claims."""
    rng = np.random.default_rng(seed)
    q = split_conformal_quantile(cal_s, alpha)
    gap = (1 - alpha) - float((tgt_s <= q).mean())
    uniq, inv = np.unique(tgt_clusters, return_inverse=True)
    cl_idx = [np.nonzero(inv == c)[0] for c in range(len(uniq))]
    g = np.empty(B)
    for b in range(B):
        qb = split_conformal_quantile(cal_s[rng.integers(0, len(cal_s), len(cal_s))], alpha)
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([cl_idx[c] for c in pick])
        g[b] = (1 - alpha) - float((tgt_s[rows] <= qb).mean())
    return {"gap": gap, "coverage": (1 - alpha) - gap, "q": float(q), "n_scenarios": int(len(uniq)),
            "ci95_cluster": [float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))]}


def run(model: str, cal_name: str) -> None:
    journal.event(f"addC_{model}", "attempt_start", f"cal={cal_name}")
    cal = load(model, cal_name)
    same = load(model, "av2test" if cal_name == "av2cal" else "nstest")
    tag = "av2gpu" if model == "av2_gpu_full" else "nsgpu"
    wm_cal = load(model, f"wm_{tag}_cal")     # unused for calibration (source cal is used, per Addendum C design);
    wm_test = load(model, f"wm_{tag}_test")   # kept so wm_cal.npz existing is verified, and for an n-check
    assert wm_test is not None, f"missing results/preds2/{model}/wm_{tag}_test.npz"
    clusters = clusters_of(wm_test)

    out = {"model": model, "target": "waymo", "n": {"cal": len(cal["scores"]), "same": len(same["scores"]),
           "wm_cal_scenarios": int(len(np.unique(clusters_of(wm_cal)))) if wm_cal is not None else None,
           "wm_test": len(wm_test["scores"]), "wm_test_scenarios": int(len(np.unique(clusters)))}}

    # H17: coverage gap on Waymo test, scenario-cluster CI
    h17_wm = boot_gap_clustered(cal["scores"], wm_test["scores"], clusters, ALPHA)
    h17_same = boot_gap(cal["scores"], same["scores"], ALPHA, B=300)
    out["H17_gap_waymo"] = h17_wm
    out["H17_gap_same_domain_control"] = h17_same
    out["H17_verdict"] = ("SUPPORTED" if h17_wm["gap"] >= 0.03 and h17_wm["ci95_cluster"][0] > 0 else
                          "REFUTED (gap<3pt or CI includes 0)")

    # H18: same source model, is the gap positive on the OTHER real target too (nuScenes for av2_gpu_full,
    # AV2 for ns_gpu_full)? Pull straight from the already-computed confirmatory run2 file.
    other_direction = "forward" if model == "av2_gpu_full" else "reverse"
    run2_path = ROOT / "results" / "run2" / f"{model}__{other_direction}.json"
    other_gap = None
    if run2_path.exists():
        other_gap = json.loads(run2_path.read_text())["H1_H4"]["0.10"]["raw"]["target"]["gap"]
    out["H18_other_target_gap"] = other_gap
    out["H18_verdict"] = ("SUPPORTED" if (other_gap is not None and other_gap > 0 and h17_wm["gap"] > 0) else
                          "gap sign differs across targets -- report as dataset-dependent, not general under-coverage")

    # H19: scene-level fix transfers to Waymo scenarios as the labelling unit
    out["H19_scene_ltt_waymo"] = h16_scene_ltt(cal["scores"], wm_test["scores"], clusters,
                                               ms=(30, 40, 50, 60), draws=200)

    # H20: normalised score reduces (not necessarily removes) the Waymo gap
    cal_n = normalized_scores(cal["pred_trajs"], cal["gt"], cal["gt_mask"], cal["pred_scale"])
    wm_n = normalized_scores(wm_test["pred_trajs"], wm_test["gt"], wm_test["gt_mask"], wm_test["pred_scale"])
    h20 = boot_gap_clustered(cal_n, wm_n, clusters, ALPHA)
    out["H20_normalized_gap_waymo"] = h20
    out["H20_verdict"] = ("reduced" if h20["gap"] < h17_wm["gap"] else "NOT reduced") + \
                         (", NOT removed" if h20["ci95_cluster"][0] > 0 else ", not distinguishable from removed")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{model}.json"
    p.write_text(json.dumps(out, indent=1))
    journal.event(f"addC_{model}", "done",
                 f"H17 gap={h17_wm['gap']:.4f} CI={h17_wm['ci95_cluster']} verdict={out['H17_verdict']}; "
                 f"H18={out['H18_verdict']}; H20={out['H20_verdict']} -> {p}")
    print(f"[addC] wrote {p}")
    print(f"  H17 waymo gap={h17_wm['gap']:.4f} (cov {h17_wm['coverage']:.4f}) CI={h17_wm['ci95_cluster']} "
          f"same-domain gap={h17_same['gap']:.4f}  -> {out['H17_verdict']}")
    print(f"  H18 other-target({other_direction}) gap={other_gap}  -> {out['H18_verdict']}")
    print(f"  H20 raw gap={h17_wm['gap']:.4f} norm gap={h20['gap']:.4f}  -> {out['H20_verdict']}")
    for m, v in out["H19_scene_ltt_waymo"]["by_m"].items():
        b = v["scene_ltt_bet"]
        print(f"  H19 m={m}: betting rate={b['rate_scene_cov_ge_nominal']:.2f} area={b['median_area_vs_oracle']:.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cal", required=True, choices=["av2cal", "nscal"])
    a = ap.parse_args()
    run(a.model, a.cal)
