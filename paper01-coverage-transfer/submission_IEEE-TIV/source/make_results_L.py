"""Macros for Addendum L (L1 sequential certification, L2 output-based monitor, L3 robust-CP baseline; L4 added when trained).
Every value is read from results/rev1/*.json, never typed. Imported by make_results_rev1.write_all()."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "rev1"


def _j(p):
    return json.loads(Path(p).read_text())


def macros_L() -> str:
    L = []
    put = lambda k, v: L.append("\\newcommand{\\%s}{%s}" % (k, v))

    p = R / "L1_sequential.json"
    if p.exists():
        d = _j(p)["pairs"].values()
        put("lOneStopMin", f"{min(x['median_stop'] for x in d):.0f}")
        put("lOneStopMax", f"{max(x['median_stop'] for x in d):.0f}")
        put("lOneFracMin", f"{100 * min(x['stop_over_fixed'] for x in d):.0f}")
        put("lOneFracMax", f"{100 * max(x['stop_over_fixed'] for x in d):.0f}")
        put("lOneAreaMin", f"{min(x['median_area_vs_oracle'] for x in d):.1f}")
        put("lOneAreaMax", f"{max(x['median_area_vs_oracle'] for x in d):.1f}")
        put("lOneValidMin", f"{min(x['p_valid'] for x in d):.3f}")

    p = R / "L2_monitor.json"
    if p.exists():
        m = _j(p)
        put("monRho", f"{m['primary_all_pairs']['S']['rho']:.2f}")
        put("monRhoShift", f"{m['shift_pairs_only']['S']['rho']:.2f}")
        put("monKS", f"{m['primary_all_pairs']['KS']['rho']:.2f}")
        put("monN", str(m["primary_all_pairs"]["S"]["n"]))
        put("monNShift", str(m["shift_pairs_only"]["S"]["n"]))
        blind = [r for r in m["pairs"] if r["model"] == "ns_gpu_full" and r["target"] == "av2test"][0]
        put("monBlindGap", f"{blind['gap_pts']:.1f}")
        put("monBlindKS", f"{blind['KS']:.2f}")
        same_ks = max(r["KS"] for r in m["pairs"] if r["kind"] == "same")
        put("monSameKSMax", f"{same_ks:.2f}")
    p = R / "L2_diagnostics.json"
    if p.exists():
        g = _j(p)
        put("monWithinKS", f"{g['within_ns']['rho_KS']:.2f}")
        put("monAucKS", f"{g['auc_flag_gap_gt3_all_pairs']['KS']:.2f}")
        put("monMAE", f"{g['loo_model_linear_MAE_pts']:.1f}")

    p = R / "L4_multi.json"
    if p.exists():
        m4 = _j(p)
        pool, wd = m4["multi"]["pooled"], m4["multi"]["worst_domain"]
        ref = m4["single_dataset_references_waymo"]
        fin = _j(R / "final_minade.json")
        put("lFourWm", f"{pool['waymo']['gap_pts']:.1f}")
        put("lFourWmLo", f"{100 * (0.9 - pool['waymo']['ci'][1]):.1f}")
        put("lFourWmHi", f"{100 * (0.9 - pool['waymo']['ci'][0]):.1f}")
        put("lFourRefAv", f"{ref['AB-AV2-s0']['gap_pts']:.1f}"); put("lFourRefNs", f"{ref['AB-NS-s0']['gap_pts']:.1f}")
        put("lFourAv", f"{abs(pool['av2_in_domain']['gap_pts']):.1f}"); put("lFourNs", f"{abs(pool['ns_in_domain']['gap_pts']):.1f}")
        put("lFourWd", f"{wd['waymo']['gap_pts']:.1f}"); put("lFourWdAv", f"{abs(wd['av2_in_domain']['gap_pts']):.1f}")
        put("lFourAdeAv", f"{fin['multi_av2ns_av2']:.3f}"); put("lFourAdeNs", f"{fin['multi_av2ns_ns']:.3f}")
        put("lFourCut", f"{100 * (1 - pool['waymo']['gap_pts'] / ref['AB-AV2-s0']['gap_pts']):.0f}")
        put("lFourBar", f"{m4['criteria']['waymo_bar_pts']:.1f}")

    p = R / "L3_robust.json"
    if p.exists():
        r3 = _j(p)
        put("robOk", str(r3["cells_succeeding"]))
        over = [c["robust"]["gap_pts"] for c in r3["cells"] if c["target"] != "Waymo"]
        under = [c["robust"]["gap_pts"] for c in r3["cells"] if c["target"] == "Waymo"]
        put("robOverMax", f"{-min(over):.0f}")
        put("robUnderMax", f"{max(under):.0f}")
    return "\n".join(L) + "\n"
