"""
Tables / macros for Addendum C (Waymo Open Motion, the third dataset), regenerated from results/addC/*.json.
Imported by make_results.py; can also be run alone. Never edit generated/*.tex by hand.

Both models here are confirmatory (gate-passing, trained after their hypotheses were registered): av2_gpu_full
(forward: AV2 -> Waymo) and ns_gpu_full (reverse: nuScenes -> Waymo). See notes/falsification.md, Addendum C.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDC = ROOT / "results" / "addC"
OUT = Path(__file__).resolve().parent / "generated"

MODELS = [("av2_gpu_full", "AV2", "AvW"), ("ns_gpu_full", "nuScenes", "NsW")]  # (file stem, source label, macro tag)


def _load(m):
    p = ADDC / f"{m}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def _ci(ci):
    return f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]"


def macros() -> str:
    L = []
    for m, _, tag in MODELS:
        d = _load(m)
        if d is None:
            continue
        h17 = d["H17_gap_waymo"]
        h20 = d["H20_normalized_gap_waymo"]
        L += [
            r"\newcommand{\wmGap%s}{%s}" % (tag, _pct(h17["gap"])),
            r"\newcommand{\wmGap%sCI}{%s}" % (tag, _ci(h17["ci95_cluster"])),
            r"\newcommand{\wmGapNorm%s}{%s}" % (tag, _pct(h20["gap"])),
            r"\newcommand{\wmNScenarios%s}{%d}" % (tag, h17["n_scenarios"]),
        ]
        if d["n"].get("wm_cal_scenarios") is not None:
            L.append(r"\newcommand{\wmNScenariosCal%s}{%d}" % (tag, d["n"]["wm_cal_scenarios"]))
        for m_, w in (("30", "Thirty"), ("40", "Forty"), ("50", "Fifty"), ("60", "Sixty")):
            b = d["H19_scene_ltt_waymo"]["by_m"].get(m_, {}).get("scene_ltt_bet")
            if b is None:
                continue
            L += [
                r"\newcommand{\wmScene%s%s}{%s}" % (tag, w, _pct(b["rate_scene_cov_ge_nominal"], 0)),
                r"\newcommand{\wmSceneArea%s%s}{%.2f}" % (tag, w, b["median_area_vs_oracle"]),
            ]
    return "\n".join(L) + "\n"


def table_waymo() -> str:
    rows = []
    for m, src, _ in MODELS:
        d = _load(m)
        if d is None:
            continue
        h17, h20 = d["H17_gap_waymo"], d["H20_normalized_gap_waymo"]
        rows.append(f"{src}$\\to$Waymo & conf. & {h17['n_scenarios']} & {_pct(h17['gap'])} & "
                    f"{_ci(h17['ci95_cluster'])} & {_pct(h20['gap'])} \\\\")
    body = "\n".join(rows) if rows else r"\multicolumn{6}{c}{no results yet} \\"
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Third-dataset replication (Waymo Open Motion v1.2.1 validation, 30/150 shards, 8730 scenarios, "
        r"scenario-level cal/test split). Both source models are confirmatory (Table~\ref{tab:models}); calibration "
        r"uses each model's own AV2 or nuScenes calibration half, as in the main result. Pre-registered as "
        r"Addendum C (H17/H18/H20) before any Waymo prediction existed.}",
        r"\label{tab:waymo}",
        r"\begin{tabular}{llrrcr}", r"\toprule",
        r"Transfer & Status & $n_\text{test}$ & gap (pt) & 95\% CI & norm.\ gap \\", r"\midrule", body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(tex)


def table_waymo_scene() -> str:
    rows = []
    for m, src, _ in MODELS:
        d = _load(m)
        if d is None:
            continue
        for k in ("30", "40", "50", "60"):
            r = d["H19_scene_ltt_waymo"]["by_m"].get(k)
            if r is None:
                continue
            b = r["scene_ltt_bet"]
            rows.append(f"{src}$\\to$Waymo & {k} & {_pct(b['rate_scene_cov_ge_nominal'], 0)} & "
                        f"{b['median_area_vs_oracle']:.2f} \\\\")
    body = "\n".join(rows) if rows else r"\multicolumn{4}{c}{no results yet} \\"
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{H19: does the scene-level betting-LTT repair (Sec.~\ref{sec:repair-scene}) transfer to Waymo "
        r"scenarios as the labelling unit (mean 3.7 agents/scenario, vs.\ nuScenes' $\sim$65/scene)? Cells: \% of "
        r"200 draws whose threshold covers $\ge 90\%$ of held-out scenarios / median area vs.\ oracle. The "
        r"pre-registered bar (rate $\ge 0.88$ at every $m\in\{30,40,50,60\}$) is met at $m=30,40$ for AV2$\to$Waymo "
        r"only; see Sec.~\ref{sec:waymo} for the honest reading of the shortfall at $m=50,60$.}",
        r"\label{tab:waymoscene}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Transfer & $m$ scenarios & rate & area/oracle \\", r"\midrule", body,
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(tex)


def write_all():
    OUT.mkdir(exist_ok=True)
    (OUT / "numbers_addC.tex").write_text(macros())
    (OUT / "table_waymo.tex").write_text(table_waymo() + "\n")
    (OUT / "table_waymo_scene.tex").write_text(table_waymo_scene() + "\n")
    return [m for m, _, _ in MODELS if _load(m)]


if __name__ == "__main__":
    print("wrote addC tables/macros for:", write_all())
