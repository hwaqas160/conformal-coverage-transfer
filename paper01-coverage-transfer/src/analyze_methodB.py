"""
Addendum H (Method B): protocol-matched label-free recalibration, AV2 -> nuScenes.
q_pm = (1-alpha) quantile of source-CALIBRATION scores computed on inputs passed through the measured protocol
transform hz2map (results/preds2/<model>/injav2cal_hz2map.npz).  No target label is used to obtain q_pm.
  (a) label-free coverage of q_pm vs q_src on the nuScenes evaluation scenes (descriptive, no guarantee claimed)
  (b) the corrected scene-level rule (Addendum D design) started from q_pm
Usage: python src/analyze_methodB.py --model av2_gpu_full --pair av2_to_nuscenes
Output: results/rev1/methodB_<pair>.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import journal  # noqa: E402
import analyze_rev1 as R  # noqa: E402
from conformal import split_conformal_quantile  # noqa: E402

ALPHA = 0.10


def boot_cov(cal_inj, tgt_s, tgt_c, q_fixed=None, B=1000, seed=0):
    """Coverage of the threshold on target scenes, resampling target SCENES (and, if q_fixed is None, the
    calibration scores i.i.d. -- AV2 clips are independent)."""
    rng = np.random.default_rng(seed)
    u, inv = np.unique(tgt_c, return_inverse=True)
    idx = [np.nonzero(inv == c)[0] for c in range(len(u))]
    out = np.empty(B)
    for b in range(B):
        q = q_fixed if q_fixed is not None else split_conformal_quantile(cal_inj[rng.integers(0, len(cal_inj), len(cal_inj))], ALPHA)
        rows = np.concatenate([idx[c] for c in rng.integers(0, len(u), len(u))])
        out[b] = (tgt_s[rows] <= q).mean()
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--draws", type=int, default=1000)
    a = ap.parse_args()
    model, cal_name, lab_names, ev_name = R.PAIRS[a.pair]
    assert model == a.model, f"{a.pair} belongs to {model}"
    journal.event(f"methodB_{a.pair}", "attempt_start", "")
    cal_s, _ = R.load_scores(model, [cal_name])
    inj_s, _ = R.load_scores(model, ["injav2cal_hz2map"])
    ev_s, ev_c = R.load_scores(model, [ev_name])
    q_src = float(split_conformal_quantile(cal_s, ALPHA))
    q_pm = float(split_conformal_quantile(inj_s, ALPHA))
    cov_src, cov_pm = float((ev_s <= q_src).mean()), float((ev_s <= q_pm).mean())
    out = {"pair": a.pair, "model": model, "alpha": ALPHA, "q_src": q_src, "q_pm": q_pm,
           "cov_src": cov_src, "cov_pm": cov_pm,
           "gap_src": (1 - ALPHA) - cov_src, "gap_pm": (1 - ALPHA) - cov_pm,
           "ci95_cov_src": boot_cov(cal_s, ev_s, ev_c), "ci95_cov_pm": boot_cov(inj_s, ev_s, ev_c),
           "gap_closed_fraction": ((1 - ALPHA) - cov_src - ((1 - ALPHA) - cov_pm)) / ((1 - ALPHA) - cov_src),
           "note": "descriptive; no guarantee is claimed for q_pm itself"}
    print(f"{a.pair}: q_src={q_src:.2f} m cov={cov_src:.3f} | q_pm={q_pm:.2f} m cov={cov_pm:.3f} | "
          f"gap {100*out['gap_src']:+.1f} -> {100*out['gap_pm']:+.1f} pt  (closed {100*out['gap_closed_fraction']:.0f}%)", flush=True)
    out["scene_rule_from_q_pm"] = R.run_pair(a.pair, draws=a.draws, q_override=q_pm)
    p = ROOT / "results" / "rev1" / f"methodB_{a.pair}.json"
    p.write_text(json.dumps(out, indent=1))
    journal.event(f"methodB_{a.pair}", "done", str(p.relative_to(ROOT)))
