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
from pathlib import Path

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover
    spearmanr = None

ROOT = Path(__file__).resolve().parent.parent
RUN2 = ROOT / "results" / "run2"
OUT = Path(__file__).resolve().parent / "generated"
OUT.mkdir(exist_ok=True)

ALPHA_KEY = "0.10"
MAIN_MODEL = "av2_cpu_v1"          # the Run-1 model; pre-registration was written against it
MAIN_DIRECTION = "forward"
K_HEADLINE = "1000"                # the label budget H5b/H5 are reported at in prose

ALL_MODELS = ["av2_cpu_v1", "av2_valsplit_v1", "av2_cpu_v2"]   # forward model zoo (H8), as they land
REVERSE_MODELS = ["ns_cpu_v1"]                                  # reverse-direction models (H9), as they land


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
    lines.append(r"\newcommand{\nModelsDone}{%d}" % len([m for m in ALL_MODELS if load(m, "forward")]))
    lines.append(r"\newcommand{\nModelsTotal}{%d}" % len(ALL_MODELS))
    lines.append(r"\newcommand{\nReverseDone}{%d}" % len([m for m in REVERSE_MODELS if load(m, "reverse")]))
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
            cj = load_cities(m)
            rho, _ = h6_rho(cj)
            rows.append({
                "model": m.replace("_", r"\_"), "direction": direction, "target": dj["target"],
                "n_cal": dj["n"]["cal"], "n_tgt": dj["n"]["target"],
                "gap_raw": pct(raw_tgt["gap"]), "ci": fmt_ci(raw_tgt["ci95"]),
                "gap_norm": pct(norm_tgt["gap"]), "reduction": f"{reduction:.0f}",
                "power": f"{power:.2f}" if power is not None else "--",
                "fa": pct(fa) if fa is not None else "--",
                "rho": f"{rho:.2f}" if rho is not None else "--",
            })
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        (r"\caption{Coverage transfer by model and direction, $\alpha=0.10$. "
         r"``red.'' = \%% reduction in gap from the label-free normalised score (H4); "
         r"``power/FA'' = audit reject probability at $k=%s$ target labels on the shifted / same-domain split (H5b); "
         r"$\rho$ = Spearman correlation between the label-free domain-monitor AUC and the per-city miscoverage (H6).}" % K_HEADLINE),
        r"\label{tab:models}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        (r"Model & Dir. & $n_\text{cal}$ & $n_\text{tgt}$ & gap (pt) & 95\% CI & norm.\ gap & red.\ (\%) & "
         r"power/FA @" + K_HEADLINE + r" & $\rho$ \\"),
        r"\midrule",
    ]
    for r in rows:
        tex.append(
            f"{r['model']} & {r['direction']} & {r['n_cal']} & {r['n_tgt']} & {r['gap_raw']} & {r['ci']} & "
            f"{r['gap_norm']} & {r['reduction']} & {r['power']}/{r['fa']} & {r['rho']} \\\\"
        )
    if not rows:
        tex.append(r"\multicolumn{10}{c}{no results yet} \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
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


if __name__ == "__main__":
    main()
