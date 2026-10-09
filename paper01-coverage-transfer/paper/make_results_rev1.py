"""
Revision R1 figures and macros, regenerated from result files (never edit generated/* by hand):

  fig_gap.pdf            coverage on the source test split vs the target, confirmatory models, 4 transfers
  fig_scene_repair.pdf   corrected scene-level repair (Addendum D): validity and region size vs labelled scenes
  numbers_rev1.tex       macros quoted in the prose

Sources: results/run2/*.json, results/addB/*.json, results/addC/*.json, results/rev1/scene_*.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "generated"
R = ROOT / "results"
ALPHA = 0.10
PAIRS = [("av2_to_nuscenes", r"AV2$\to$nuSc."), ("av2_to_waymo", r"AV2$\to$Waymo"),
         ("ns_to_av2", r"nuSc.$\to$AV2"), ("ns_to_waymo", r"nuSc.$\to$Waymo")]
# second architecture (Addendum E): same recommended-budget treatment, kept out of fig_gap/fig_scene_repair's
# AutoBot-only panels (see fig_scene_repair) to avoid crowding, but in the same `scene` dict / macro loop so
# recMWfNs/recMWfWm etc. are generated uniformly.
PAIRS_WF = [("wayformer_to_nuscenes", r"Wayf.$\to$nuSc."), ("wayformer_to_waymo", r"Wayf.$\to$Waymo")]
MACRO_TAG = {"av2_to_nuscenes": "AvNs", "av2_to_waymo": "AvWm", "ns_to_av2": "NsAv", "ns_to_waymo": "NsWm",
            "wayformer_to_nuscenes": "WfNs", "wayformer_to_waymo": "WfWm"}

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "lines.linewidth": 1.1})


def _j(p):
    return json.loads(Path(p).read_text())


def gap_rows():
    """(label, same-domain coverage, its CI, target coverage, its CI); CIs are on the gap, converted to coverage."""
    fwd = _j(R / "run2/av2_gpu_full__forward.json")["H1_H4"]["0.10"]["raw"]
    rev = _j(R / "run2/ns_gpu_full__reverse.json")["H1_H4"]["0.10"]["raw"]
    b10 = _j(R / "addB/av2_gpu_full__forward.json")["B10_target_clustered"]
    wa = _j(R / "addC/av2_gpu_full.json")
    wn = _j(R / "addC/ns_gpu_full.json")
    cov = lambda g: 1 - ALPHA - g
    ci = lambda c: (cov(c[1]), cov(c[0]))
    return [
        (PAIRS[0][1], fwd["same"]["coverage"], ci(fwd["same"]["ci95"]), cov(b10["gap"]), ci(b10["ci95_cluster"])),
        (PAIRS[1][1], fwd["same"]["coverage"], ci(fwd["same"]["ci95"]), wa["H17_gap_waymo"]["coverage"],
         ci(wa["H17_gap_waymo"]["ci95_cluster"])),
        (PAIRS[2][1], rev["same"]["coverage"], ci(rev["same"]["ci95"]), rev["target"]["coverage"], ci(rev["target"]["ci95"])),
        (PAIRS[3][1], rev["same"]["coverage"], ci(rev["same"]["ci95"]), wn["H17_gap_waymo"]["coverage"],
         ci(wn["H17_gap_waymo"]["ci95_cluster"])),
    ]


def fig_gap():
    rows = gap_rows()
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    x = range(len(rows))
    w = 0.36
    for i, (lab, cs, cis, ct, cit) in enumerate(rows):
        ax.bar(i - w / 2, cs, w, color="#56B4E9", edgecolor="#0072B2", lw=0.5, label="source test split" if i == 0 else None)
        ax.bar(i + w / 2, ct, w, color="#D55E00", edgecolor="#7a3000", lw=0.5, hatch="////",
               label="target dataset" if i == 0 else None)
        for xx, c, ci in ((i - w / 2, cs, cis), (i + w / 2, ct, cit)):
            ax.errorbar(xx, c, yerr=[[c - ci[0]], [ci[1] - c]], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.5)
        ax.text(i + w / 2, ct - 0.018, f"{100 * ct:.0f}", ha="center", va="top", fontsize=5.5, color="white")
    ax.axhline(1 - ALPHA, ls="--", color="k", lw=0.7)
    ax.text(-0.45, 1 - ALPHA + 0.004, "nominal", fontsize=5.5, ha="left", va="bottom")
    ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(0.6, 0.97); ax.set_ylabel("coverage")
    ax.legend(loc="lower center", frameon=False, ncol=2, bbox_to_anchor=(0.5, 0.99), borderaxespad=0.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_gap.pdf")
    plt.close(fig)


def _wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def fig_traj():
    """Coverage by manoeuvre, confirmatory AV2 model: AV2 test vs nuScenes Boston vs nuScenes Singapore."""
    import numpy as np
    b = _j(R / "addB/av2_gpu_full__forward.json")["B11_driving_side"]["by_type"]
    types = [t for t in ("stationary", "straight", "straight_right", "straight_left", "right_turn", "left_turn")
             if t in b and b[t].get("boston") is not None and b[t].get("singapore") is not None]
    labels = {"stationary": "stationary", "straight": "straight", "straight_right": "lane ch. R",
              "straight_left": "lane ch. L", "right_turn": "right turn", "left_turn": "left turn"}
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    x = np.arange(len(types)); w = 0.27
    series = [("same", "n_same", "AV2 test", "#0072B2", ""), ("boston", "n_boston", "nuSc. Boston", "#E69F00", "...."),
              ("singapore", "n_singapore", "nuSc. Singapore", "#D55E00", "////")]
    for j, (k, nk, lab, col, hatch) in enumerate(series):
        v = np.array([b[t][k] for t in types]); n = np.array([b[t][nk] for t in types])
        lo, hi = zip(*[_wilson(p, nn) for p, nn in zip(v, n)])
        ax.bar(x + (j - 1) * w, v, w, color=col, edgecolor="0.15", lw=0.4, hatch=hatch, label=lab,
               yerr=[v - np.array(lo), np.array(hi) - v], error_kw={"elinewidth": 0.5, "capsize": 1.2})
    ax.axhline(1 - ALPHA, color="k", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels([labels[t] for t in types])
    ax.set_ylim(0.2, 1.02); ax.set_ylabel("coverage")
    ax.legend(loc="lower center", frameon=False, ncol=3, bbox_to_anchor=(0.5, 0.99), borderaxespad=0.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_traj_conf.pdf")
    plt.close(fig)


def recommended(d):
    """Addendum D4 rule (as applied in the outcome log): smallest m with median area <= 2.0x oracle and
    an infinite threshold in <= 5% of draws."""
    for m in sorted(d["by_m"], key=int):
        b = d["by_m"][m]["bet"]
        if b["median_area_vs_oracle"] is not None and b["median_area_vs_oracle"] <= 2.0 and b["frac_infinite"] <= 0.05:
            return int(m), b
    return None, None


def fig_scene_repair(scene):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.85))
    d = scene["av2_to_nuscenes"]
    ms = sorted(map(int, d["by_m"]))
    styles = [("bet", "scene LTT, betting", "o", "-", "#0072B2"), ("hb", "scene LTT, HB", "s", "--", "#009E73"),
              ("agent_cor", "agent-level C-or-R", "^", ":", "#D55E00"), ("hier_sub1", "one agent / scene", "v", "-.", "#E69F00")]
    for k, lab, mk, ls, c in styles:
        a1.plot(ms, [d["by_m"][str(m)][k]["p_valid"] for m in ms], marker=mk, ms=2.6, ls=ls, color=c, label=lab)
    a1.axhline(1 - 0.10, color="k", lw=0.6, ls=(0, (1, 1)))
    a1.set_xscale("log"); a1.set_xticks([20, 50, 100, 300]); a1.set_xticklabels(["20", "50", "100", "300"])
    a1.set_ylim(0, 1.04); a1.set_xlabel("labelled scenes $m$"); a1.set_ylabel(r"P[$R(\hat q)\leq\alpha$]")
    a1.set_title("(a) validity, AV2$\\to$nuSc.", fontsize=6.5)
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=5,
               handlelength=2.2, columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))
    marks = ["o", "s", "^", "D", "P", "X"]; lss = ["-", "--", ":", "-.", "-", "--"]
    cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
    all_pairs = PAIRS + [p for p in PAIRS_WF if p[0] in scene]
    for (pk, plab), mk, ls, col in zip(all_pairs, marks, lss, cols):
        dd = scene[pk]
        xs, ys, hollow = [], [], []
        for m in sorted(map(int, dd["by_m"])):
            b = dd["by_m"][str(m)]["bet"]
            if b["median_area_vs_oracle"] is None:
                continue
            xs.append(m); ys.append(b["median_area_vs_oracle"]); hollow.append(b["frac_infinite"] > 0.05)
        line, = a2.plot(xs, ys, ls=ls, color=col, lw=0.8, label=plab)
        for x_, y_, h in zip(xs, ys, hollow):
            a2.plot(x_, y_, marker=mk, ms=2.8, color=col, mfc="white" if h else col, ls="none")
    a2.axhline(2.0, color="k", lw=0.6, ls=(0, (1, 1)))
    a2.set_xscale("log"); a2.set_xticks([60, 100, 200, 300]); a2.set_xticklabels(["60", "100", "200", "300"])
    a2.set_xlabel("labelled scenes $m$"); a2.set_ylabel("area / oracle")
    a2.set_title("(b) betting LTT, region size", fontsize=6.5)
    a2.legend(loc="upper right", frameon=False, fontsize=4.3, handlelength=1.8, labelspacing=0.25)
    for a in (a1, a2):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_scene_repair.pdf")
    plt.close(fig)


def macros(scene):
    L = []
    put = lambda k, v: L.append(r"\newcommand{\%s}{%s}" % (k, v))
    vals = [b["p_valid"] for d in scene.values() for row in d["by_m"].values() for kk, b in row.items() if kk == "bet"]
    put("betValidMin", f"{min(vals):.2f}")
    recs = []
    for pk, _ in PAIRS:
        d = scene[pk]
        m, b = recommended(d)
        tag = MACRO_TAG[pk]
        put(f"recM{tag}", m if m else "--")
        put(f"recArea{tag}", f"{b['median_area_vs_oracle']:.1f}" if b else "--")
        put(f"recQ{tag}", f"{b['median_q_finite_m']:.1f}" if b else "--")
        put(f"qSrc{tag}", f"{d['q_src']:.1f}")
        put(f"qOracle{tag}", f"{d['q_oracle']['0.10']:.1f}")
        put(f"nLabScenes{tag}", d["n_label_scenes"])
        recs.append(m)
    put("recMMin", min(r for r in recs if r)); put("recMMax", max(r for r in recs if r))
    put("infAtEightyAvNs", f"{100 * scene['av2_to_nuscenes']['by_m']['80']['bet']['frac_infinite']:.0f}")
    put("infAtHundredAvNs", f"{100 * scene['av2_to_nuscenes']['by_m']['100']['bet']['frac_infinite']:.0f}")
    ns = scene["av2_to_nuscenes"]["by_m"]
    put("agentCorNsForty", f"{100 * ns['40']['agent_cor']['p_valid']:.0f}")
    put("agentCorNsThreeHundred", f"{100 * ns['300']['agent_cor']['p_valid']:.0f}")
    put("hierNsThreeHundred", f"{100 * ns['300']['hier_sub1']['p_valid']:.0f}")
    put("hbAreaNsTwoHundred", f"{ns['200']['hb']['median_area_vs_oracle']:.1f}")
    put("betAreaNsTwoHundred", f"{ns['200']['bet']['median_area_vs_oracle']:.1f}")
    others = [scene[p]["by_m"]["300"]["agent_cor"]["p_valid"] for p in ("av2_to_waymo", "ns_to_waymo", "ns_to_av2")]
    put("agentCorOtherMin", f"{100 * min(others):.0f}")
    tgt = [r[3] for r in gap_rows()]
    wf_run2, wf_addc = R / "run2/av2_wayformer__forward.json", R / "addC/av2_wayformer.json"
    if wf_run2.exists() and wf_addc.exists():
        tgt = tgt + [_j(wf_run2)["H1_H4"]["0.10"]["raw"]["target"]["coverage"],
                    1 - _j(wf_addc)["H17_gap_waymo"]["gap"]]
    put("covTgtMin", f"{100 * min(tgt):.0f}")
    put("covTgtMax", f"{100 * max(tgt):.0f}")
    return "\n".join(L) + "\n"


def macros_wf(scene) -> str:
    """Second architecture (Wayformer, Addendum E). Gap/reweighting/manoeuvre numbers from run2+addB json
    (same quantities already reported for the AutoBot confirmatory models); scene-repair from `scene` (rev1,
    corrected design, same as the AutoBot pairs)."""
    L = []
    put = lambda k, v: L.append(r"\newcommand{\%s}{%s}" % (k, v))
    pct = lambda x: f"{100 * x:.1f}"

    fwd = _j(R / "run2/av2_wayformer__forward.json")
    h = fwd["H1_H4"]["0.10"]["raw"]
    hn = fwd["H1_H4"]["0.10"]["norm"]
    put("minADESixWayformer", "0.928")   # measured directly on last.ckpt, Addendum E outcome log (not in a ckpt filename)
    put("gapSameWayformer", pct(h["same"]["gap"]))
    put("gapRawWayformer", pct(h["target"]["gap"]))
    put("gapRawWayformerCI", f"[{pct(h['target']['ci95'][0])}, {pct(h['target']['ci95'][1])}]")
    put("gapNormWayformer", pct(hn["target"]["gap"]))
    put("normReductionWayformer", f"{100 * (1 - hn['target']['gap'] / h['target']['gap']):.0f}")
    h3 = fwd["H3_label_free"]
    put("rewWayformer", f"{100 * (h3['weighted']['gap'] - h3['uncorrected']['gap']):+.1f}")
    audit = fwd["audit"]
    put("auditPowerWayformer", f"{audit['target']['reject_prob'].get('1000'):.2f}")
    put("auditFaWayformer", f"{100 * audit['same_domain']['reject_prob'].get('1000'):.0f}")

    wa = _j(R / "addC/av2_wayformer.json")
    put("wmGapWf", pct(wa["H17_gap_waymo"]["gap"]))
    put("wmGapWfCI", f"[{pct(wa['H17_gap_waymo']['ci95_cluster'][0])}, {pct(wa['H17_gap_waymo']['ci95_cluster'][1])}]")
    put("wmGapNormWf", pct(wa["H20_normalized_gap_waymo"]["gap"]))

    b = _j(R / "addB/av2_wayformer__forward.json")
    tt = lambda dom, t: b["B7"]["raw"][dom]["trajectory_type"][t]["coverage"]
    put("dropRightWf", f"{100 * (tt('same', 'right_turn') - tt('target', 'right_turn')):.0f}")
    put("dropStraightWf", f"{100 * (tt('same', 'straight') - tt('target', 'straight')):.1f}")
    put("designEffWf", f"{b['B10_target_clustered']['design_effect']:.1f}")
    put("sgExtraRightWf", f"{100 * b['B11_driving_side']['by_type']['right_turn']['drop_diff_sg_minus_bos']:.0f}")

    # agent-level C-or-R: SAME nuScenes target, compare severity directly against av2_gpu_full (AutoBot)
    ab = scene["av2_to_nuscenes"]; wf = scene["wayformer_to_nuscenes"]
    put("rSrcAutobot", f"{100 * ab['R_src_on_eval']:.1f}")
    put("rSrcWayformer", f"{100 * wf['R_src_on_eval']:.1f}")
    put("agentCorWfForty", f"{100 * wf['by_m']['40']['agent_cor']['p_valid']:.0f}")
    put("agentCorWfThreeHundred", f"{100 * wf['by_m']['300']['agent_cor']['p_valid']:.0f}")
    for pk, _ in PAIRS_WF:
        d = scene[pk]
        m, b = recommended(d)
        tag = MACRO_TAG[pk]
        put(f"recM{tag}", m if m else "--")
        put(f"recArea{tag}", f"{b['median_area_vs_oracle']:.1f}" if b else "--")
    inj = _j(R / "addB/inject_av2_wayformer.json")["injections"]       # Addendum E-2
    put("wfInjHz", f"{100 * inj['hz2']['effect_pt']:.1f}")
    put("wfInjHzFrac", f"{100 * inj['hz2']['fraction_of_G']:.0f}")
    put("wfInjBoth", f"{100 * inj['hz2map']['effect_pt']:.1f}")
    put("wfInjBothFrac", f"{100 * inj['hz2map']['fraction_of_G']:.0f}")
    put("wfInjMapFrac", f"{100 * inj['map']['fraction_of_G']:.0f}")
    return "\n".join(L) + "\n"


def macros_methodB() -> str:
    """Addendum H (Method B): protocol-matched label-free recalibration, AV2 -> nuScenes, both architectures."""
    L = []
    put = lambda k, v: L.append(r"\newcommand{\%s}{%s}" % (k, v))
    for tag, pair in (("Ab", "av2_to_nuscenes"), ("Wf", "wayformer_to_nuscenes")):
        p = R / f"rev1/methodB_{pair}.json"
        if not p.exists():
            continue
        d = _j(p)
        put(f"mbCovSrc{tag}", f"{100 * d['cov_src']:.0f}")
        put(f"mbCovPm{tag}", f"{100 * d['cov_pm']:.0f}")
        put(f"mbGapSrc{tag}", f"{100 * d['gap_src']:.1f}")
        put(f"mbGapPm{tag}", f"{100 * d['gap_pm']:+.1f}")
        put(f"mbClosed{tag}", f"{100 * d['gap_closed_fraction']:.0f}")
        put(f"mbQSrc{tag}", f"{d['q_src']:.1f}")
        put(f"mbQPm{tag}", f"{d['q_pm']:.1f}")
        m, b = recommended(d["scene_rule_from_q_pm"])
        put(f"mbRecM{tag}", m if m else "--")
    return "\n".join(L) + "\n"


def write_all():
    OUT.mkdir(exist_ok=True)
    scene = {pk: _j(R / f"rev1/scene_{pk}.json") for pk, _ in PAIRS + PAIRS_WF
             if (R / f"rev1/scene_{pk}.json").exists()}
    fig_gap()
    fig_traj()
    if all(pk in scene for pk, _ in PAIRS):
        fig_scene_repair(scene)
        macro_text = macros(scene)
        if all(pk in scene for pk, _ in PAIRS_WF):
            macro_text += macros_wf(scene)
        macro_text += macros_methodB()
        (OUT / "numbers_rev1.tex").write_text(macro_text)
    return sorted(scene)


if __name__ == "__main__":
    print("rev1 figures/macros for:", write_all())
