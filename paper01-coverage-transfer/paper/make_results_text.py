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


STATS = R / "rev1" / "text_stats.json"


def _raw_stats():
    """Values that need the prediction archives / checkpoint names (not in the repo): computed once and cached to
    results/rev1/text_stats.json, which IS committed, so a fresh clone regenerates the same macros."""
    ade = [_best_minade6(m) for m in ("av2_cpu_v1", "av2_valsplit_v1", "av2_cpu_v2")]
    ns = np.load(PRED / "av2_gpu_full/ns.npz", allow_pickle=True)
    sid = _scene(ns["scenario_id"]); _, cnt = np.unique(sid, return_counts=True)
    nsc = lambda f: int(len(np.unique(_scene(np.load(PRED / "ns_gpu_full" / f, allow_pickle=True)["scenario_id"]))))
    rows, scen = 0, set()
    for n in ("wmcal", "wmtest"):
        w = np.load(PRED / "av2_wayformer" / f"{n}.npz", allow_pickle=True)
        rows += len(w["scenario_id"]); scen |= set(map(str, w["scenario_id"]))
    return {"ade_expl_min": min(ade), "ade_expl_max": max(ade), "ade_conf": _best_minade6("av2_gpu_full"),
            "ns_n_agents": int(len(sid)), "ns_n_scenes": int(len(cnt)), "ns_mean": float(cnt.mean()),
            "ns_median": float(np.median(cnt)), "ns_cal": nsc("nscal.npz"), "ns_test": nsc("nstest.npz"),
            "wm_agents_per_scenario": rows / len(scen)}


def _stats():
    if (PRED / "av2_gpu_full/ns.npz").exists():
        st = _raw_stats(); STATS.write_text(json.dumps(st, indent=1))
        return st
    return _j(STATS)


def macros_text() -> str:
    L = []
    put = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))

    # ---- reduced-budget AutoBot models (Table models) ------------------------------------------------------
    expl_fwd = ["av2_cpu_v1", "av2_valsplit_v1", "av2_cpu_v2"]
    gaps = [100 * _j(R / f"run2/{m}__forward.json")["H1_H4"][KEY]["raw"]["target"]["gap"] for m in expl_fwd]
    put("explGapMin", f"{min(gaps):.1f}"); put("explGapMax", f"{max(gaps):.1f}")
    st = _stats()
    put("explAdeMin", f"{st['ade_expl_min']:.2f}"); put("explAdeMax", f"{st['ade_expl_max']:.2f}")

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
    fin = _j(R / "rev1/final_minade.json")
    put("confAccRelPct", f"{100 * abs(fin['av2_gpu_full'] - REF_ACC) / REF_ACC:.1f}")
    put("nsAccRelPct", f"{100 * (fin['ns_gpu_full'] - 1.21) / 1.21:.0f}")
    put("seedOneAde", f"{fin['av2_gpu_full_seed1']:.3f}")

    # ---- nuScenes / Waymo dataset statistics -----------------------------------------------------------------
    put("nsNAgents", str(st["ns_n_agents"])); put("nsNScenes", str(st["ns_n_scenes"]))
    put("nsAgentsMean", f"{st['ns_mean']:.0f}"); put("nsAgentsMedian", f"{st['ns_median']:.0f}")
    put("nsNScenesCal", str(st["ns_cal"])); put("nsNScenesTest", str(st["ns_test"]))
    put("wmAgentsPerScenario", f"{st['wm_agents_per_scenario']:.1f}")

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


def macros_audit_fix() -> str:
    """Agent-level C-or-R validity by transfer, audit rejection on shift-free splits, normalised score on Wayformer,
    and the descriptive planner-relevance numbers (Addendum F; agent-weighted, no guarantee)."""
    L = []
    put = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))
    sc = {p: _j(R / f"rev1/scene_{p}.json") for p in ("av2_to_waymo", "ns_to_av2", "wayformer_to_waymo", "ns_to_waymo")}
    ok = [100 * r["agent_cor"]["p_valid"] for p in ("av2_to_waymo", "ns_to_av2", "wayformer_to_waymo")
          for r in sc[p]["by_m"].values() if "agent_cor" in r]
    put("agentCorSmallMin", f"{min(ok):.0f}")
    nw = sc["ns_to_waymo"]["by_m"]
    put("agentCorNsWmForty", f"{100 * nw['40']['agent_cor']['p_valid']:.0f}")
    put("agentCorNsWmThreeHundred", f"{100 * nw['300']['agent_cor']['p_valid']:.0f}")
    put("rSrcNsWm", f"{100 * sc['ns_to_waymo']['R_src_on_eval']:.1f}")
    runs = [("av2_gpu_full", "forward"), ("av2_gpu_full_seed1", "forward"), ("av2_wayformer", "forward"),
            ("av2_wayformer_seed1", "forward"), ("ns_gpu_full", "reverse")]
    fa = [_j(R / f"run2/{m}__{d}.json")["audit"]["same_domain"] for m, d in runs]
    put("auditFaAllMin", f"{100 * min(a['reject_prob']['1000'] for a in fa):.0f}")
    put("auditFaAllMax", f"{100 * max(a['reject_prob']['1000'] for a in fa):.0f}")
    put("auditSameGapMax", f"{100 * max(a['true_gap'] for a in fa):.1f}")
    wf = _j(R / "run2/av2_wayformer__forward.json")["H1_H4"][KEY]
    put("normReductionWf", f"{100 * (1 - wf['norm']['target']['gap'] / wf['raw']['target']['gap']):.0f}")
    f = _j(R / "addF/addF.json")
    put("missAgentsAvNs", f"{100 * f['av2_to_nuscenes']['M_agent_weighted']:.0f}")
    put("missAgentsAvWm", f"{100 * f['av2_to_waymo']['M_agent_weighted']:.0f}")
    b = _j(R / "addB/av2_gpu_full__forward.json")["B11_driving_side"]["by_type"]
    put("missRightSg", f"{100 * (1 - b['right_turn']['singapore']):.0f}")
    put("missRightBos", f"{100 * (1 - b['right_turn']['boston']):.0f}")
    put("missStraightBos", f"{100 * (1 - b['straight']['boston']):.0f}")
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


def macros_methodsAB() -> str:
    """Addendum K: worst-domain (A) and manoeuvre-conditional (B) calibration, six cells."""
    p = R / "rev1/methodsAB.json"
    if not p.exists():
        return ""
    d = _j(p); cells = d["cells"]
    wm = [c["A_worst_domain"]["gap_pts"] for c in cells if c["target"] == "Waymo"]
    easy = [c["A_worst_domain"]["gap_pts"] for c in cells if c["target"] != "Waymo"]
    red = [100 * c["B_mondrian"]["gap_reduction_fraction"] for c in cells]
    L = []
    put = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))
    put("abAok", str(d["A_cells_succeeding"])); put("abBok", str(d["B_cells_succeeding"]))
    put("abWmGapMin", f"{min(wm):.0f}"); put("abWmGapMax", f"{max(wm):.0f}")
    put("abEasyOverMax", f"{-min(easy):.0f}")
    put("abBRedMed", f"{np.median(red):.0f}")
    return "\n".join(L) + "\n"
