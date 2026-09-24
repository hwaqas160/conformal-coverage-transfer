"""
Tables / figures / macros for the Addendum-B results, regenerated from results/addB/*.json.
Imported by make_results.py; can also be run alone.  Never edit generated/*.tex by hand.

Every table carries a `status` column: models listed in EXPLORATORY were run before the corresponding hypothesis
was registered on them (or were the models on which a design decision was made), CONFIRMATORY models are the
competitive, gate-passing models trained after registration.  A model appears in exactly one class.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ADDB = ROOT / "results" / "addB"
OUT = Path(__file__).resolve().parent / "generated"

EXPLORATORY = ["av2_cpu_v1", "av2_valsplit_v1", "av2_cpu_v2"]
CONFIRMATORY = ["av2_gpu_full"]              # + gated GPU models as they are added (Wayformer, nuScenes-source, ...)
ALL = EXPLORATORY + CONFIRMATORY


def _tex(m: str) -> str:
    return m.replace("_", r"\_")


def _status(m: str) -> str:
    return "conf." if m in CONFIRMATORY else "expl."


def _load(m: str, direction="forward"):
    p = ADDB / f"{m}__{direction}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x, d=0):
    return "--" if x is None else f"{100 * x:.{d}f}"


# ------------------------------------------------------------------------------------------------ tables
def table_repair(models=ALL, ks=("250", "1000")) -> str:
    """Agent-level labels (i.i.d.): rate of covering >= 1-alpha, and area vs oracle, for direct / PAC / C-or-R."""
    rows = []
    for m in models:
        d = _load(m)
        if d is None:
            continue
        for k in ks:
            r = d["B5_target"]["by_k"].get(k)
            if r is None:
                continue
            cells = " & ".join(f"{_pct(r[x]['rate_cov_ge_nominal'])} / {r[x]['median_area_vs_oracle']:.2f}"
                               for x in ("direct", "pac", "cor"))
            rows.append(f"{_tex(m)} & {_status(m)} & {k} & {cells} \\\\")
    body = "\n".join(rows) if rows else r"\multicolumn{6}{c}{no results yet} \\"
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Repair with $k$ labelled target scenes, labels exchangeable. Cells: \% of label draws whose threshold "
        r"covers $\ge 90\%$ of the held-out target scenes / median region area relative to the oracle threshold. "
        r"C-or-R = certify-or-recalibrate ($\delta=0.1$).}", r"\label{tab:repair}",
        r"\begin{tabular}{llrccc}", r"\toprule",
        r"Model & Set & $k$ & direct SCP & PAC & C-or-R \\", r"\midrule", body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def table_scene(models=ALL, ms=("30", "40", "60")) -> str:
    """Whole-scene labels: the realistic case.  agent-level C-or-R vs scene-level LTT (HB and betting)."""
    rows = []
    for m in models:
        d = _load(m)
        if d is None or "H16_scene_ltt" not in d:
            continue
        for k in ms:
            r = d["H16_scene_ltt"]["by_m"].get(k)
            if r is None:
                continue

            def cell(x, area=True):
                a = r[x]["median_area_vs_oracle"]
                return f"{_pct(r[x]['rate_scene_cov_ge_nominal'])}" + (f" / {a:.1f}" if area and np.isfinite(a) else (" / $\\infty$" if area else ""))
            rows.append(f"{_tex(m)} & {_status(m)} & {k} & {cell('direct', False)} & {cell('agent_cor', False)} & "
                        f"{cell('scene_ltt')} & {cell('scene_ltt_bet')} \\\\")
    body = "\n".join(rows) if rows else r"\multicolumn{7}{c}{no results yet} \\"
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Repair when whole \emph{scenes} are labelled (nuScenes: $\sim$65 correlated agents per scene). "
        r"Cells: \% of draws whose threshold covers $\ge 90\%$ of held-out scenes (scene-averaged) / median area vs oracle. "
        r"Agent-level guarantees fail under clustering; scene-level Learn-then-Test restores them, "
        r"and the betting $p$-value is far tighter than Hoeffding--Bentkus.}", r"\label{tab:scene}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrcccc}", r"\toprule",
        r"Model & Set & $m$ scenes & direct & C-or-R (agent) & LTT (HB) & LTT (betting) \\", r"\midrule", body,
        r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


def table_inject(models=ALL) -> str:
    rows = []
    for m in models:
        p = ADDB / f"inject_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        inj = d["injections"]

        def c(k):
            v = inj.get(k)
            return "--" if not v else f"{100 * v['effect_pt']:+.1f} ({_pct(v['fraction_of_G'])}\\%)"
        comp = d["composition"]
        rows.append(f"{_tex(m)} & {_status(m)} & {100 * d['G_real_gap']:+.1f} & {c('hz2')} & {c('map')} & {c('hz2map')} & "
                    f"{100 * comp['effect_pt']:+.1f} ({_pct(comp['fraction_of_G'])}\\%) \\\\")
    body = "\n".join(rows) if rows else r"\multicolumn{7}{c}{no results yet} \\"
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{Where does the loss come from? Coverage lost (points; share of the real gap $G$ in parentheses) when "
        r"one measured AV2--nuScenes difference is injected into AV2 test inputs: 2\,Hz history, lane-graph dropout, both, "
        r"and re-weighting AV2 scenes to nuScenes' scene mix. Excess annotation noise and map-point density were measured "
        r"and found absent (nuScenes is smoother; equal density), so they are not injected.}", r"\label{tab:inject}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrcccc}", r"\toprule",
        r"Model & Set & $G$ (pt) & 2\,Hz & lanes & both & scene mix \\", r"\midrule", body, r"\bottomrule", r"\end{tabular}}", r"\end{table}"])


# ------------------------------------------------------------------------------------------------ figures
def fig_traj_type(models=None):
    """Coverage by manoeuvre type: same-domain vs target, one panel per model (Wilson CIs from B7)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [m for m in (models or ALL) if _load(m)]
    if not models:
        return
    types = ["stationary", "straight", "straight_right", "straight_left", "right_turn", "left_turn"]
    fig, axes = plt.subplots(1, len(models), figsize=(7.0, 2.5), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        d = _load(m)["B7"]["raw"]
        x = np.arange(len(types)); w = 0.38
        for j, (dom, lab) in enumerate((("same", "AV2 (same domain)"), ("target", "nuScenes"))):
            v = [d[dom]["trajectory_type"][t]["coverage"] for t in types]
            lo = [d[dom]["trajectory_type"][t]["ci95"][0] for t in types]; hi = [d[dom]["trajectory_type"][t]["ci95"][1] for t in types]
            ax.bar(x + (j - 0.5) * w, v, w, yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], capsize=1.5, label=lab)
        ax.axhline(0.9, color="k", lw=0.7, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([t.replace("_", " ") for t in types], rotation=40, ha="right", fontsize=6.5)
        ax.set_title(m, fontsize=7); ax.set_ylim(0.2, 1.02)
    axes[0][0].set_ylabel("coverage at $\\alpha=0.1$"); axes[0][0].legend(fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig_traj_type.pdf"); plt.close(fig)


def table_cluster(models=ALL) -> str:
    """H1 gap with agent-level vs scene-clustered bootstrap CI (B10)."""
    rows = []
    for m in models:
        d = _load(m)
        if d is None:
            continue
        b = d["B10_target_clustered"]
        rows.append(f"{_tex(m)} & {_status(m)} & {100 * b['gap']:+.1f} & [{100 * b['ci95_iid'][0]:.1f}, {100 * b['ci95_iid'][1]:.1f}] & "
                    f"[{100 * b['ci95_cluster'][0]:.1f}, {100 * b['ci95_cluster'][1]:.1f}] & {b['design_effect']:.1f} \\\\")
    body = "\n".join(rows) if rows else r"\multicolumn{6}{c}{no results yet} \\"
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{The coverage gap on nuScenes with agent-level vs.\ scene-clustered bootstrap intervals (138 scenes, $\sim$65 agents each). "
        r"The design effect is the variance inflation caused by clustering.}", r"\label{tab:cluster}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{llrccr}", r"\toprule",
        r"Model & Set & gap (pt) & 95\% CI (agents) & 95\% CI (scenes) & design eff. \\", r"\midrule", body, r"\bottomrule",
        r"\end{tabular}}", r"\end{table}"])


def macros(models=EXPLORATORY) -> str:
    """Every number quoted in the Addendum-B prose, as min--max over `models` (percent unless stated)."""
    ds = {m: _load(m) for m in models if _load(m)}
    inj = {m: json.loads((ADDB / f"inject_{m}.json").read_text()) for m in ds if (ADDB / f"inject_{m}.json").exists()}
    L = []

    def rng(name, vals, fmt="{:.0f}", scale=100.0):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            lo, hi = min(vals) * scale, max(vals) * scale
            L.append(r"\newcommand{\%sMin}{%s}" % (name, fmt.format(lo))); L.append(r"\newcommand{\%sMax}{%s}" % (name, fmt.format(hi)))
        else:
            L.append(r"\newcommand{\%sMin}{--}" % name); L.append(r"\newcommand{\%sMax}{--}" % name)

    g = lambda d, *path: _dig(d, path)
    rng("corDirect", [g(d, "B5_target", "by_k", "1000", "direct", "rate_cov_ge_nominal") for d in ds.values()])
    rng("corPAC", [g(d, "B5_target", "by_k", "1000", "pac", "rate_cov_ge_nominal") for d in ds.values()])
    rng("corCOR", [g(d, "B5_target", "by_k", "1000", "cor", "rate_cov_ge_nominal") for d in ds.values()])
    rng("corCORarea", [g(d, "B5_target", "by_k", "1000", "cor", "median_area_vs_oracle") for d in ds.values()], "{:.2f}", 1.0)
    for m_, w in (("40", "Forty"), ("60", "Sixty")):        # macro names cannot contain digits
        rng(f"sceneBet{w}", [g(d, "H16_scene_ltt", "by_m", m_, "scene_ltt_bet", "rate_scene_cov_ge_nominal") for d in ds.values()])
        rng(f"sceneBetArea{w}", [g(d, "H16_scene_ltt", "by_m", m_, "scene_ltt_bet", "median_area_vs_oracle") for d in ds.values()], "{:.1f}", 1.0)
        rng(f"sceneAgent{w}", [g(d, "H16_scene_ltt", "by_m", m_, "agent_cor", "rate_scene_cov_ge_nominal") for d in ds.values()])
        rng(f"sceneHBArea{w}", [g(d, "H16_scene_ltt", "by_m", m_, "scene_ltt", "median_area_vs_oracle") for d in ds.values()], "{:.1f}", 1.0)
    rng("aciFast", [g(d, "B6_target", "by_gamma", "0.05", "running_cov_labels_to_stay_within_tol_median") for d in ds.values()], "{:.0f}", 1.0)
    rng("aciSlow", [g(d, "B6_target", "by_gamma", "0.005", "running_cov_labels_to_stay_within_tol_median") for d in ds.values()], "{:.0f}", 1.0)
    rng("aciFrozen", [g(d, "B6_target", "by_gamma", "0.01", "frozen_after_k", "1000", "rate_cov_ge_nominal") for d in ds.values()])
    for key, nm in (("hz2", "injHz"), ("map", "injMap"), ("hz2map", "injBoth")):
        rng(nm, [(v["injections"].get(key) or {}).get("fraction_of_G") for v in inj.values()])
    rng("injComp", [v["composition"]["effect_pt"] for v in inj.values()], "{:.1f}", 100.0)
    rng("injCompFrac", [v["composition"]["fraction_of_G"] for v in inj.values()])
    tt = lambda d, dom, t: d["B7"]["raw"][dom]["trajectory_type"][t]["coverage"]
    rng("dropRight", [tt(d, "same", "right_turn") - tt(d, "target", "right_turn") for d in ds.values()], "{:.0f}", 100.0)
    rng("dropStraight", [tt(d, "same", "straight") - tt(d, "target", "straight") for d in ds.values()], "{:.1f}", 100.0)
    rng("dropStraightRight", [tt(d, "same", "straight_right") - tt(d, "target", "straight_right") for d in ds.values()], "{:.0f}", 100.0)
    rng("sgExtraRight", [g(d, "B11_driving_side", "by_type", "right_turn", "drop_diff_sg_minus_bos") for d in ds.values()], "{:.0f}", 100.0)
    rng("sgExtraStraight", [g(d, "B11_driving_side", "by_type", "straight", "drop_diff_sg_minus_bos") for d in ds.values()], "{:.1f}", 100.0)
    rng("designEff", [g(d, "B10_target_clustered", "design_effect") for d in ds.values()], "{:.1f}", 1.0)
    rng("clusterLo", [g(d, "B10_target_clustered", "ci95_cluster")[0] if g(d, "B10_target_clustered", "ci95_cluster") else None for d in ds.values()], "{:.1f}", 100.0)
    L.append(r"\newcommand{\nAddB}{%d}" % len(ds))
    return "\n".join(L) + "\n"


def _dig(d, path):
    for p in path:
        if d is None or p not in d:
            return None
        d = d[p]
    return d


def write_all():
    OUT.mkdir(exist_ok=True)
    (OUT / "numbers_addB.tex").write_text(macros())
    (OUT / "table_cluster.tex").write_text(table_cluster() + "\n")
    (OUT / "table_repair.tex").write_text(table_repair() + "\n")
    (OUT / "table_scene.tex").write_text(table_scene() + "\n")
    (OUT / "table_inject.tex").write_text(table_inject() + "\n")
    fig_traj_type()
    have = [m for m in ALL if _load(m)]
    return have


if __name__ == "__main__":
    print("wrote addB tables/figs for:", write_all())
