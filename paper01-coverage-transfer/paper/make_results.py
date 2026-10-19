"""
Regenerate every number and table in the paper from results/run2/*.json. Never edit generated/*.tex by hand.

Usage:
  F:\\CLAUDE\\AI1\\shared\\envs\\unitraj\\Scripts\\python.exe paper\\make_results.py

Writes:
  paper/generated/numbers.tex        -- \\newcommand macros used inline in prose (main model = av2_cpu_v1)
  paper/generated/table_models.tex   -- one row per model actually present in results/run2/ (H8 robustness)
  paper/generated/table_cities.tex   -- H6/H7 city-level summary, one row per model actually present

Models/directions not yet run are simply absent from the tables -- nothing is interpolated or guessed. The
script prints a status line for each model/direction it did NOT find, so it's obvious from stdout what's
still pending.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

ROOT = Path(__file__).resolve().parent.parent
RUN2 = ROOT / "results" / "run2"
OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)

ALPHA_KEY = "0.10"
MAIN_MODEL = "av2_cpu_v1"          # the Run-1 model; pre-registration was written against it
MAIN_DIRECTION = "forward"
K_HEADLINE = "1000"                # the label budget H5b/H5 are reported at in prose

ALL_MODELS = ["av2_cpu_v1", "av2_valsplit_v1", "av2_cpu_v2", "av2_gpu_full"]   # forward model zoo (H8), as they land
REVERSE_MODELS = ["ns_cpu_v1", "ns_gpu_full"]                                  # reverse-direction models (H9), as they land
CONF_MODEL = "av2_gpu_full"          # the gate-passing, pre-registered confirmatory forward model
CONF_REVERSE_MODEL = "ns_gpu_full"   # the gate-passing, pre-registered confirmatory reverse model


def best_minade6(model):
    """Lowest val/minADE6 among the model's saved checkpoints (Lightning's own monitored metric, filename
    epochNN-minADEx.xxx.ckpt) -- the number the pre-registered accuracy gate is checked against. NOT the same as
    base_model['minADE5_2Hz_same'] in run2 json, which is a different (5-mode, 2Hz-subsampled) diagnostic metric."""
    d = ROOT / "results" / "ckpts" / model
    vals = [float(m.group(1)) for f in d.glob("epoch*.ckpt") if (m := re.search(r"minADE([\d.]+)\.ckpt$", f.name))]
    if vals:
        return min(vals)
    cache = ROOT / "results" / "rev1" / "ckpt_minade.json"   # committed copy, for a fresh clone without checkpoints
    return json.loads(cache.read_text()).get(model) if cache.exists() else None


def load(model, direction):
    p = RUN2 / f"{model}__{direction}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_cities(model):
    p = RUN2 / f"cities__{model}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return f"{100*x:.1f}"


def fmt_ci(ci):
    return f"[{pct(ci[0])}, {pct(ci[1])}]"


def h6_rho(cities_json):
    if cities_json is None or spearmanr is None:
        return None, None
    pairs = cities_json["H6"]["pairs"]
    if len(pairs) < 4:
        return None, None
    aucs = [p["auc"] for p in pairs]
    gaps = [p["abs_gap"] for p in pairs]
    rho, pval = spearmanr(aucs, gaps)
    return rho, pval


ESTIMATOR_LABELS = {
    "source_only": "source-only (no target labels)",
    "direct": "direct (target-only SCP)",
    "pooled": "pooled",
    "shrink_k0=100": r"shrinkage ($k_0{=}100$)",
    "shrink_k0=500": r"shrinkage ($k_0{=}500$)",
    "shrink_k0=2000": r"shrinkage ($k_0{=}2000$)",
}
PLOT_ESTIMATORS = ["source_only", "direct", "pooled", "shrink_k0=500"]


def make_label_budget_figure(model, direction="forward"):
    """H5: p(|coverage error| < tol) vs k, one line per estimator. Returns k* for 'direct', or None."""
    dj = load(model, direction)
    if dj is None or plt is None:
        return None
    lb = dj["label_budget"].get(ALPHA_KEY)
    if lb is None:
        return None
    ks = lb["ks"]
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for est in PLOT_ESTIMATORS:
        if est not in lb["estimators"]:
            continue
        ys = [lb["estimators"][est][str(k)]["p_within_tol"] for k in ks]
        ax.plot(ks, ys, marker="o", markersize=3, label=ESTIMATOR_LABELS[est])
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"target-labelled scenes $k$")
    ax.set_ylabel(f"P(|coverage err| < {lb['tol']})")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title(model, fontsize=9)  # plain text (matplotlib, not TeX) -- no LaTeX escaping here
    fig.tight_layout()
    fig.savefig(OUT / f"fig_label_budget_{model}.pdf")
    plt.close(fig)
    return lb["k_star"].get("direct")


def main():
    present, missing = [], []
    for m in ALL_MODELS:
        (present if load(m, MAIN_DIRECTION) else missing).append(("forward", m))
    for m in REVERSE_MODELS:
        (present if load(m, "reverse") else missing).append(("reverse", m))
    for d, m in missing:
        print(f"[pending] {m} / {d}: results/run2/{m}__{d}.json not found yet")

    # ---- headline macros (main model only) --------------------------------------------------
    d = load(MAIN_MODEL, MAIN_DIRECTION)
    lines = []
    if d is not None:
        h = d["H1_H4"][ALPHA_KEY]
        raw_same, raw_tgt = h["raw"]["same"], h["raw"]["target"]
        norm_tgt = h["norm"]["target"]
        reduction = 100 * (1 - norm_tgt["gap"] / raw_tgt["gap"]) if raw_tgt["gap"] else 0.0
        audit_power = d["audit"]["target"]["reject_prob"].get(K_HEADLINE)
        audit_fa = d["audit"]["same_domain"]["reject_prob"].get(K_HEADLINE)
        lines += [
            r"\newcommand{\gapRawMain}{%s}" % pct(raw_tgt["gap"]),
            r"\newcommand{\gapRawMainCI}{%s}" % fmt_ci(raw_tgt["ci95"]),
            r"\newcommand{\gapSameMain}{%s}" % pct(raw_same["gap"]),
            r"\newcommand{\gapNormMain}{%s}" % pct(norm_tgt["gap"]),
            r"\newcommand{\normReductionMain}{%.0f}" % reduction,
            r"\newcommand{\nCalMain}{%d}" % d["n"]["cal"],
            r"\newcommand{\nTargetMain}{%d}" % d["n"]["target"],
        ]
        if audit_power is not None:
            lines.append(r"\newcommand{\auditPowerK}{%.2f}" % audit_power)
        if audit_fa is not None:
            lines.append(r"\newcommand{\auditFalseAlarmK}{%s}" % pct(audit_fa))
        h3 = d["H3_label_free"]
        lines += [
            r"\newcommand{\gapUncorrMain}{%s}" % pct(h3["uncorrected"]["gap"]),
            r"\newcommand{\gapWeightedMain}{%s}" % pct(h3["weighted"]["gap"]),
            r"\newcommand{\removalFracMain}{%.0f}" % (100 * d["H2"]["label_free_reweighting_removal_fraction"]),
        ]
    # ---- confirmatory headline macros (gate-passing GPU models, both directions) -------------
    for tag, model, direction in (("Conf", CONF_MODEL, "forward"), ("ConfRev", CONF_REVERSE_MODEL, "reverse"),
                                  ("RevExpl", "ns_cpu_v1", "reverse")):
        dc = load(model, direction)
        if dc is None:
            continue
        h = dc["H1_H4"][ALPHA_KEY]
        raw_same, raw_tgt = h["raw"]["same"], h["raw"]["target"]
        norm_tgt = h["norm"]["target"]
        reduction = 100 * (1 - norm_tgt["gap"] / raw_tgt["gap"]) if raw_tgt["gap"] else 0.0
        lines += [
            r"\newcommand{\gapRaw%s}{%s}" % (tag, pct(raw_tgt["gap"])),
            r"\newcommand{\gapRaw%sCI}{%s}" % (tag, fmt_ci(raw_tgt["ci95"])),
            r"\newcommand{\gapSame%s}{%s}" % (tag, pct(raw_same["gap"])),
            r"\newcommand{\gapNorm%s}{%s}" % (tag, pct(norm_tgt["gap"])),
            r"\newcommand{\normReduction%s}{%.0f}" % (tag, reduction),
            r"\newcommand{\nCal%s}{%d}" % (tag, dc["n"]["cal"]),
            r"\newcommand{\nTarget%s}{%d}" % (tag, dc["n"]["target"]),
        ]
        b6 = best_minade6(model)
        if b6 is not None:
            lines.append(r"\newcommand{\minADESix%s}{%.3f}" % (tag, b6))
    lines.append(r"\newcommand{\nModelsDone}{%d}" % len([m for m in ALL_MODELS if load(m, "forward")]))
    lines.append(r"\newcommand{\nModelsTotal}{%d}" % len(ALL_MODELS))
    lines.append(r"\newcommand{\nReverseDone}{%d}" % len([m for m in REVERSE_MODELS if load(m, "reverse")]))
    k_star = make_label_budget_figure(MAIN_MODEL, MAIN_DIRECTION)
    if k_star is not None:
        lines.append(r"\newcommand{\kStarDirect}{%d}" % k_star)
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")

    # ---- per-model table (H8 robustness), forward + reverse rows whenever present ------------
    rows = []
    for direction, models in (("forward", ALL_MODELS), ("reverse", REVERSE_MODELS)):
        for m in models:
            dj = load(m, direction)
            if dj is None:
                continue
            h = dj["H1_H4"][ALPHA_KEY]
            raw_tgt, norm_tgt = h["raw"]["target"], h["norm"]["target"]
            reduction = 100 * (1 - norm_tgt["gap"] / raw_tgt["gap"]) if raw_tgt["gap"] else float("nan")
            power = dj["audit"]["target"]["reject_prob"].get(K_HEADLINE)
            fa = dj["audit"]["same_domain"]["reject_prob"].get(K_HEADLINE)
            h3 = dj["H3_label_free"]
            rows.append({
                "model": m.replace("_", r"\_"), "status": "conf." if m in (CONF_MODEL, CONF_REVERSE_MODEL) else "expl.",
                "direction": "fwd" if direction == "forward" else "rev",
                "gap_raw": pct(raw_tgt["gap"]),
                "ci": f"{pct(raw_tgt['ci95'][0])}--{pct(raw_tgt['ci95'][1])}",
                "gap_norm": pct(norm_tgt["gap"]),
                "rew": f"{100 * (h3['weighted']['gap'] - h3['uncorrected']['gap']):+.1f}",
                "power": f"{power:.2f}" if power is not None else "--",
                "fa": f"{100 * fa:.0f}" if fa is not None else "--",
            })
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Coverage gap by model ($\alpha=0.10$, points; target nuScenes for fwd., AV2 for rev.). "
         r"conf.: gated full-data model; expl.: reduced-budget model used to design the analyses. norm.: gap of "
         r"the normalised score. rew.: change in gap from covariate reweighting (positive = worse). audit: reject "
         r"probability / false-alarm \%% at $k=%s$ labels.}" % K_HEADLINE),
        r"\label{tab:models}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\columnwidth}{!}{\begin{tabular}{lllrcrrc}",
        r"\toprule",
        r"Model & St. & Dir. & gap & 95\% CI & norm. & rew. & audit \\",
        r"\midrule",
    ]
    for r in sorted(rows, key=lambda r: (r["status"] != "conf.", r["direction"])):
        tex.append(f"{r['model']} & {r['status']} & {r['direction']} & {r['gap_raw']} & {r['ci']} & "
                   f"{r['gap_norm']} & {r['rew']} & {r['power']}/{r['fa']} \\\\")
    if not rows:
        tex.append(r"\multicolumn{8}{c}{no results yet} \\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    (OUT / "table_models.tex").write_text("\n".join(tex) + "\n")

    # ---- H2/H3 reweighting table: does covariate reweighting repair the gap? ------------------
    rw_rows = []
    for direction, models in (("forward", ALL_MODELS), ("reverse", REVERSE_MODELS)):
        for m in models:
            dj = load(m, direction)
            if dj is None:
                continue
            h3 = dj["H3_label_free"]
            rw_rows.append({
                "model": m.replace("_", r"\_"), "direction": direction,
                "uncorr": pct(h3["uncorrected"]["gap"]), "weighted": pct(h3["weighted"]["gap"]),
                "removal": f"{100*dj['H2']['label_free_reweighting_removal_fraction']:.0f}",
            })
    tex2 = [
        r"\begin{table}[t]",
        r"\centering",
        (r"\caption{Does covariate-shift reweighting repair the gap? ``uncorr.'' / ``weighted'' are the "
         r"label-free-score coverage gap (points) before / after importance-weighting the calibration scores by "
         r"the label-free-feature likelihood ratio; ``removal'' is the \% of the gap's cross-city variance the "
         r"weights explain, negative meaning reweighting makes the fit worse than the unweighted baseline.}"),
        r"\label{tab:reweight}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Dir. & uncorr.\ & weighted & removal (\%) \\",
        r"\midrule",
    ]
    for r in rw_rows:
        tex2.append(f"{r['model']} & {r['direction']} & {r['uncorr']} & {r['weighted']} & {r['removal']} \\\\")
    if not rw_rows:
        tex2.append(r"\multicolumn{5}{c}{no results yet} \\")
    tex2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_reweight.tex").write_text("\n".join(tex2) + "\n")

    print(f"wrote {OUT/'numbers.tex'} ({len(lines)} macros), {OUT/'table_models.tex'} ({len(rows)} rows), "
          f"{OUT/'table_reweight.tex'} ({len(rw_rows)} rows)")

    # Addendum-B tables/figures (repair, scene-level repair, injection, conditional coverage)
    import make_results_addB
    print("addB tables/figs for:", make_results_addB.write_all())

    # Addendum-C tables/macros (Waymo, third dataset)
    import make_results_addC
    print("addC tables/macros for:", make_results_addC.write_all())

    # Revision R1 figures/macros (gap bars, corrected scene-level repair)
    import make_results_rev1
    print("rev1 figures/macros for:", make_results_rev1.write_all())


if __name__ == "__main__":
    main()
