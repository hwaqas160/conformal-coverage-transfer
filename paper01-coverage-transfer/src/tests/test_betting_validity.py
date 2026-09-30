"""
Addendum D5 regression tests for the scene-level betting p-value (revision R1).

  (a) adversarial scenes with losses in {0,1} (single-agent scenes) never drive the betting capital negative;
  (b) exact-superpopulation validity: scenes are drawn i.i.d. from a KNOWN distribution whose true scene-averaged
      miscoverage R(q) is analytic, so P[R(q_hat) <= alpha] >= 1 - delta - 2 SE can be checked exactly over 2000 draws
      for m in {20, 40, 60, 100}, for a source threshold that is (i) badly wrong, (ii) borderline, (iii) already valid;
  (c) mutation check: restoring the old bet cap (0.5/alpha) must make (a) fail.

Run:  python src/tests/test_betting_validity.py
"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import solutions  # noqa: E402
from solutions import scene_ltt_threshold, wsr_pvalue  # noqa: E402

ALPHA, DELTA = 0.10, 0.10
A, B, MU, SIG = 0.0, 0.35, 1.0, 0.45            # ln(scene difficulty) ~ N(A,B^2); ln(score | scene) ~ N(MU + ln d, SIG^2)


def true_R(q: float) -> float:
    """Exact scene-averaged miscoverage E_j[P(score > q | scene j)] = P(score > q) for the marginal log-normal."""
    return 1.0 if q <= 0 else float(norm.sf((np.log(q) - (A + MU)) / sqrt(B * B + SIG * SIG)))


def q_at(R: float) -> float:
    return float(np.exp(A + MU + sqrt(B * B + SIG * SIG) * norm.isf(R)))


def draw_scenes(rng, m: int, singleton_frac: float = 0.25):
    """m scenes; a share are single-agent (hard case for the bet cap), the rest have 2..90 agents."""
    out = []
    for _ in range(m):
        d = np.exp(rng.normal(A, B))
        n = 1 if rng.random() < singleton_frac else int(rng.integers(2, 91))
        out.append(np.exp(rng.normal(MU + np.log(d), SIG, n)))
    return out


def test_a_no_negative_capital(rng) -> bool:
    ok = True
    for _ in range(300):
        L = (rng.random(int(rng.integers(1, 120))) < rng.choice([0.0, 0.02, 0.1, 0.3, 0.9])).astype(float)
        try:
            wsr_pvalue(L, ALPHA)                     # the function asserts capital >= 0 at every step
        except AssertionError:
            ok = False
    return ok


def test_b_exact_validity(rng, draws=2000, ms=(20, 40, 60, 100, 200)) -> bool:
    ok = True
    for label, R_src in (("q_src far too small (R=.20)", 0.20), ("q_src borderline (R=.11)", 0.11),
                         ("q_src already valid (R=.05)", 0.05)):
        q_src = q_at(R_src)
        for m in ms:
            hit = np.zeros(2)
            areas = []
            fin = np.zeros(2)
            for _ in range(draws):
                lab = draw_scenes(rng, m)
                for j, pv in enumerate(("hb", "wsr")):
                    q, _ = scene_ltt_threshold(q_src, lab, ALPHA, DELTA, pval=pv)
                    hit[j] += true_R(q) <= ALPHA
                    fin[j] += np.isfinite(q)
                    if pv == "wsr" and np.isfinite(q):
                        areas.append(q / q_at(ALPHA))
            if m == 200:                       # informational usefulness row (Addendum D-2), fewer draws in the loop below
                pass
            rate = hit / draws
            se = np.sqrt(rate * (1 - rate) / draws)
            need = 1 - DELTA - 2 * np.sqrt((1 - DELTA) * DELTA / draws)
            good = bool((rate >= need).all())
            ok &= good
            print(f"  {label:30s} m={m:3d}: P[R<=alpha]  HB={rate[0]:.3f}  betting={rate[1]:.3f} (+-{se[1]:.3f})  "
                  f"need>={need:.3f}  finite HB/bet={fin[0]/draws:.2f}/{fin[1]/draws:.2f}  median betting area vs oracle="
                  f"{(np.median(np.square(areas)) if areas else float('nan')):.2f}  {'OK' if good else 'FAIL'}")
    return ok


def test_c_mutation(rng) -> bool:
    """Restore the old cap (lam <= 0.5/alpha, i.e. BET_CAP = 0.5(1-alpha)/alpha): test (a) must now fail."""
    saved = solutions.BET_CAP
    solutions.BET_CAP = 0.5 * (1 - ALPHA) / ALPHA
    try:
        broken = not test_a_no_negative_capital(np.random.default_rng(1))
    finally:
        solutions.BET_CAP = saved
    return broken


if __name__ == "__main__":
    rng = np.random.default_rng(20260930)
    r = {}
    r["a_no_negative_capital"] = test_a_no_negative_capital(rng)
    print(f"[D5a] adversarial {{0,1}} losses, no negative capital: {'OK' if r['a_no_negative_capital'] else 'FAIL'}")
    r["c_mutation_detected"] = test_c_mutation(rng)
    print(f"[D5c] old cap makes (a) fail (mutation detected): {'OK' if r['c_mutation_detected'] else 'FAIL'}")
    print("[D5b] exact superpopulation validity (2000 draws each):")
    r["b_exact_validity"] = test_b_exact_validity(rng)
    print("\n" + "  ".join(f"{k}:{'OK' if v else 'FAIL'}" for k, v in r.items()))
    sys.exit(0 if all(r.values()) else 1)
