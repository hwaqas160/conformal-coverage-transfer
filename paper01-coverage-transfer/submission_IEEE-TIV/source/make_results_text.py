"""Macros for numbers that used to be typed by hand in the running text of Setup/Methods/Results.

Every value is computed from a result file (or from the prediction archives), never typed. Imported by
make_results_rev1.write_all(); output goes to generated/numbers_rev1.tex together with the other rev1 macros.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
PRED = R / "preds2"
KEY = "0.10"
REF_ACC = 0.85  # UniTraj's published AutoBot minADE6 on AV2 (arXiv:2403.15098, Table 15)


def _j(p):
    return json.loads(Path(p).read_text())


def _best_minade6(model):
    d = R / "ckpts" / model
    d = d if d.exists() else R / ("archive_" + model.split("av2_")[-1]) / "ckpts"  # valsplit_v1 was archived
    vals = [float(m.group(1)) for f in d.glob("epoch*.ckpt")
            if (m := re.search(r"minADE([\d.]+)\.ckpt$", f.name))]
    return min(vals) if vals else None


def _scene(ids):
    return np.array([(re.search(r"scene-\d+", str(s)) or [str(s)])[0] for s in ids])


def macros_text() -> str:
    L = []
    put = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))

    # ---- reduced-budget AutoBot models (Table models) ------------------------------------------------------
    expl_fwd = ["av2_cpu_v1", "av2_valsplit_v1", "av2_cpu_v2"]
    gaps = [100 * _j(R / f"run2/{m}__forward.json")["H1_H4"][KEY]["raw"]["target"]["gap"] for m in expl_fwd]
    put("explGapMin", f"{min(gaps):.1f}"); put("explGapMax", f"{max(gaps):.1f}")
    ade = [_best_minade6(m) for m in expl_fwd]
    put("explAdeMin", f"{min(ade):.2f}"); put("explAdeMax", f"{max(ade):.2f}")

    red = []
    for m, d in [(m, "forward") for m in expl_fwd] + [("ns_cpu_v1", "reverse")]:
        h = _j(R / f"run2/{m}__{d}.json")["H1_H4"][KEY]
        red.append(100 * (1 - h["norm"]["target"]["gap"] / h["raw"]["target"]["gap"]))
    put("explNormBestRed", f"{max(red):.0f}"); put("explNormWorstInc", f"{-min(red):.0f}")

    # ---- audit on the confirmatory models ------------------------------------------------------------------
    pw, fa = [], []
    for m, d in (("av2_gpu_full", "forward"), ("ns_gpu_full", "reverse")):
        a = _j(R / f"run2/{m}__{d}.json")["audit"]
        pw.append(a["target"]["reject_prob"]["1000"]); fa.append(a["same_domain"]["reject_prob"]["1000"])
    put("auditPowerConfMin", f"{min(pw):.2f}"); put("auditFaConfMax", f"{100 * max(fa):.0f}")

    # ---- label-free shift monitor: largest Spearman over every model with a city analysis -------------------
    rho = [_j(p)["H6"]["spearman_auc_vs_abs_gap"] for p in sorted((R / "run2").glob("cities__*.json"))
           if "dev_ep11" not in p.name]
    put("monitorRhoMax", f"{max(rho):.2f}")

    # ---- confirmatory accuracy relative to the published reference -------------------------------------------
    b = _best_minade6("av2_gpu_full")
    put("confAccRelPct", f"{100 * abs(b - REF_ACC) / REF_ACC:.1f}")

    # ---- nuScenes / Waymo dataset statistics -----------------------------------------------------------------
    ns = np.load(PRED / "av2_gpu_full/ns.npz", allow_pickle=True)
    sid = _scene(ns["scenario_id"]); u, cnt = np.unique(sid, return_counts=True)
    put("nsNAgents", str(len(sid))); put("nsNScenes", str(len(u)))
    put("nsAgentsMean", f"{cnt.mean():.0f}"); put("nsAgentsMedian", f"{np.median(cnt):.0f}")
    put("nsNScenesCal", str(len(np.unique(_scene(np.load(PRED / "ns_gpu_full/nscal.npz", allow_pickle=True)["scenario_id"])))))
    put("nsNScenesTest", str(len(np.unique(_scene(np.load(PRED / "ns_gpu_full/nstest.npz", allow_pickle=True)["scenario_id"])))))
    wm_rows, wm_scen = 0, set()
    for n in ("wmcal", "wmtest"):
        w = np.load(PRED / "av2_wayformer" / f"{n}.npz", allow_pickle=True)
        wm_rows += len(w["scenario_id"]); wm_scen |= set(map(str, w["scenario_id"]))
    put("wmAgentsPerScenario", f"{wm_rows / len(wm_scen):.1f}")

    # ---- measured input differences used by the injections ---------------------------------------------------
    ip = _j(R / "addB/inject_params.json")
    put("lanesMedianAv", f"{ip['lanes_median']['av2']:.0f}"); put("lanesMedianNs", f"{ip['lanes_median']['ns']:.0f}")
    put("noiseSigmaAv", f"{ip['av2']['sigma_m']:.2f}"); put("noiseSigmaNs", f"{ip['ns']['sigma_m']:.2f}")

    # ---- detection budget from the closed form of Eq. (budget) -----------------------------------------------
    from statistics import NormalDist
    z = NormalDist().inv_cdf
    a_, d_, lvl, beta = 0.1, 0.034, 0.05, 0.1
    k = ((z(1 - lvl) * math.sqrt(a_ * (1 - a_)) + z(1 - beta) * math.sqrt((a_ + d_) * (1 - a_ - d_))) / d_) ** 2
    put("auditBudgetK", f"{round(k, -1):.0f}")
    put("agentsForTwoPoints", f"{round(1.96 ** 2 * a_ * (1 - a_) / 0.02 ** 2, -2):.0f}")
    return "\n".join(L) + "\n"


def macros_wf_seed() -> str:
    """Addendum I: second Wayformer training seed."""
    p = R / "rev1/wf_seed1.json"
    if not p.exists():
        return ""
    d = _j(p); a, b = d["av2_wayformer"], d["av2_wayformer_seed1"]
    L = []
    put = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))
    put("wfSeedGapOne", f"{100 * b['gap_nuscenes']:.1f}"); put("wfSeedGapZero", f"{100 * a['gap_nuscenes']:.1f}")
    put("wfSeedWmOne", f"{100 * b['gap_waymo']:.1f}"); put("wfSeedWmZero", f"{100 * a['gap_waymo']:.1f}")
    put("wfSeedAde", f"{b['val_minADE6_last']:.3f}")
    return "\n".join(L) + "\n"
