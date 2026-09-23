"""
Synthetic validation of src/solutions.py, with ground truth known BY CONSTRUCTION:

  (1) H4 mechanics: if the shift acts through the difficulty the model's own scale can see, the normalised
      score transfers (gap ~ 0) while the raw score does not; if the shift is INVISIBLE to the scale, the
      normalised score does not fix it (honest negative control).
  (2) H5 mechanics: 'direct' is unbiased with sd ~ sqrt(a(1-a)/k); 'pooled' is biased toward the source;
      shrinkage sits in between.
  (3) H6 mechanics: domain AUC ~0.5 with no shift, >0.7 with a clear shift.

Run:  python src/tests/test_solutions_synthetic.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from conformal import SplitConformal  # noqa: E402
from solutions import (normalized_scores, region_area_proxy, label_budget, domain_auc, audit_power)  # noqa: E402

K, T = 6, 60


def make_arrays(n, difficulty, err_mult, scale_sees_difficulty, rng):
    """gt random walk; best mode = gt + N(0, sigma_i) with sigma_i = err_mult*difficulty; other modes far away."""
    gt = np.cumsum(rng.normal(0, 0.3, (n, T, 2)), axis=1).astype(np.float32)
    sig = err_mult * difficulty                                            # true error scale per sample
    good = gt + rng.normal(0, 1, (n, T, 2)) * sig[:, None, None]
    bad = gt[:, None] + rng.normal(0, 8, (n, K - 1, T, 2))
    pred = np.concatenate([good[:, None], bad], axis=1).astype(np.float32)
    # model's predicted scale: tracks difficulty if it can "see" it, else constant
    u_good = 0.03 * (difficulty if scale_sees_difficulty else np.ones(n)) * rng.lognormal(0, 0.15, n)
    u = np.concatenate([u_good[:, None], np.full((n, K - 1), 0.05)], axis=1).astype(np.float32)
    return pred, gt, np.ones((n, T), bool), u


def cover_gap(cal_s, tgt_s, alpha=0.10, seeds=5):
    g = []
    for sd in range(seeds):
        r = np.random.default_rng(sd); idx = r.permutation(len(cal_s))[: len(cal_s) // 2]
        g.append((1 - alpha) - SplitConformal(alpha).calibrate(cal_s[idx]).evaluate(tgt_s)["coverage"])
    return float(np.mean(g))


def test_h4(rng):
    n = 6000
    dS = rng.lognormal(0, 0.35, n)                       # source difficulty
    # A) shift visible to the model's scale: target scenes are harder AND u knows it
    dT = rng.lognormal(0.45, 0.35, n)
    cs = make_arrays(n, dS, 1.0, True, rng); ct = make_arrays(n, dT, 1.0, True, rng)
    raw_gap = cover_gap(np.array(_raw(cs)), np.array(_raw(ct)))
    nrm_gap = cover_gap(normalized_scores(*cs[:2], cs[2], cs[3]), normalized_scores(ct[0], ct[1], ct[2], ct[3]))
    print(f"[H4 visible shift ] raw gap={raw_gap:+.3f}   normalised gap={nrm_gap:+.3f}   (expect raw >> normalised ~ 0)")
    ok1 = raw_gap > 0.10 and abs(nrm_gap) < 0.03
    # B) invisible shift: same difficulty distribution, but target errors are 1.5x larger and u cannot see it
    ct2 = make_arrays(n, dS, 1.5, False, rng); cs2 = make_arrays(n, dS, 1.0, False, rng)
    raw2 = cover_gap(np.array(_raw(cs2)), np.array(_raw(ct2)))
    nrm2 = cover_gap(normalized_scores(cs2[0], cs2[1], cs2[2], cs2[3]), normalized_scores(ct2[0], ct2[1], ct2[2], ct2[3]))
    print(f"[H4 invisible shift] raw gap={raw2:+.3f}   normalised gap={nrm2:+.3f}   (expect BOTH large: normalisation is not magic)")
    ok2 = raw2 > 0.10 and nrm2 > 0.10
    # efficiency proxy is finite and sensible
    q = 3.0; a = region_area_proxy(q, n, K, cs[3])
    return ok1 and ok2 and np.all(np.isfinite(a))


def _raw(c):
    from conformal import nonconformity_scores
    return nonconformity_scores(c[0], c[1], c[2])


def test_h5(rng):
    src = rng.lognormal(1.0, 0.6, 5000)
    tgt = rng.lognormal(1.0, 0.6, 9000) * 1.15           # target scores 15% larger (a pure scale shift)
    res = label_budget(src, tgt, ks=(50, 250, 1000), alpha=0.10, draws=150, seed=1)
    e = res["estimators"]
    for nm in ("source_only", "direct", "pooled", "shrink_k0=500"):
        v = e[nm]["250"]
        print(f"[H5 k=250] {nm:15s} signed err={v['mean_signed_err']:+.4f}  sd={v['sd']:.4f}  P(|err|<=.02)={v['p_within_tol']:.2f}")
    d250 = e["direct"]["250"]; so = e["source_only"]["250"]; pl = e["pooled"]["250"]
    sd_theory = np.sqrt(0.1 * 0.9 / 250)
    ok = (abs(d250["mean_signed_err"]) < 0.01                       # direct ~ unbiased (small +bias from finite-sample rule)
          and 0.5 * sd_theory < d250["sd"] < 1.6 * sd_theory        # sd ~ binomial theory
          and so["mean_signed_err"] < -0.03                         # source-only undercovers under the shift
          and so["mean_signed_err"] < pl["mean_signed_err"] < d250["mean_signed_err"] + 1e-9)  # pooled in between
    print(f"      theory sd(direct,k=250)={sd_theory:.4f}   k*={res['k_star']['direct']}")
    return bool(ok)


def test_h6(rng):
    a = rng.normal(0, 1, (3000, 5)); b = rng.normal(0, 1, (3000, 5)); c = rng.normal(0.9, 1, (3000, 5))
    auc0, auc1 = domain_auc(a, b), domain_auc(a, c)
    print(f"[H6] AUC no-shift={auc0:.3f} (expect ~0.5)   shifted={auc1:.3f} (expect >0.7)")
    return 0.45 < auc0 < 0.56 and auc1 > 0.70


def _theory_reject(k, alpha_true, alpha=0.10, level=0.05):
    """Exact P(audit rejects) when the threshold's true miss probability is alpha_true."""
    from scipy.stats import binom
    m = np.arange(k + 1)
    rej = binom.sf(m - 1, k, alpha) <= level
    return float(binom.pmf(m, k, alpha_true)[rej].sum())


def test_audit(rng):
    """
    (a) the simulator matches the exact binomial theory for the realised threshold (function correctness);
    (b) MARGINAL false-alarm rate, averaged over fresh source-calibration draws, is close to the nominal level
        (the realised coverage of any single calibrated threshold fluctuates ~0.4 pt, so a single q can be
        flagged more often than 5% at large k -- that is the audit doing its job, not a false alarm);
    (c) power increases with k under a real shift.
    """
    from conformal import split_conformal_quantile
    pool_same = rng.lognormal(1.0, 0.6, 40000)
    shifted = rng.lognormal(1.0, 0.6, 40000) * 1.15
    # (a)
    q = split_conformal_quantile(rng.lognormal(1.0, 0.6, 5000), 0.10)
    at = float((shifted > q).mean())
    sim = audit_power(q, shifted, ks=(250, 1000), draws=1500, seed=3)["reject_prob"]
    th = {k: _theory_reject(k, at) for k in (250, 1000)}
    print(f"[H5b audit] (a) power sim vs exact theory: k=250 {sim['250']:.3f} vs {th[250]:.3f} | k=1000 {sim['1000']:.3f} vs {th[1000]:.3f}")
    ok_a = abs(sim["250"] - th[250]) < 0.05 and abs(sim["1000"] - th[1000]) < 0.05
    # (b) marginal false alarm over 60 fresh calibration draws, k=1000
    fa = []
    for i in range(60):
        qi = split_conformal_quantile(rng.lognormal(1.0, 0.6, 5000), 0.10)
        fa.append(audit_power(qi, pool_same, ks=(1000,), draws=60, seed=100 + i)["reject_prob"]["1000"])
    print(f"            (b) marginal false-alarm rate (k=1000, same distribution) = {np.mean(fa):.3f} (expect <= ~0.08)")
    ok_b = np.mean(fa) < 0.09
    print(f"            (c) power under shift: k=250 {sim['250']:.2f} -> k=1000 {sim['1000']:.2f}")
    return bool(ok_a and ok_b and sim["1000"] > sim["250"] > 0.3)


def test_cor(rng):
    """
    Addendum B5 (certify-or-recalibrate) + the PAC quantile it rests on, against their stated guarantees:
      (a) q_pac: true coverage >= 1-alpha in >= 1-delta of draws (continuous scores, known distribution);
      (b) C-or-R under a real shift: same guarantee, and it must RECALIBRATE (source threshold is invalid);
      (c) certification needs SLACK: with no shift, a source threshold calibrated at exactly alpha has miss
          probability ~= alpha, so an upper confidence bound <= alpha is (almost) never achievable -- expect ~0
          certifications.  A source threshold calibrated with slack (alpha_src = alpha/2) is certifiable at large k.
          [Corrected 2026-09-23: the first version of this test wrongly expected (c) to certify at nominal alpha;
          the procedure was right, the expectation was not.  Recorded in journal/JOURNAL.md.]
      (d) plain direct SCP at the same k is only marginally valid: rate well below 1-delta (~50-60%).
    True coverage is computed exactly from the known lognormal CDF, not from a finite evaluation set.
    """
    from scipy.stats import lognorm
    from conformal import split_conformal_quantile
    from solutions import q_pac, certify_or_recalibrate, q_direct
    alpha, delta, k, draws = 0.10, 0.10, 500, 2000
    src_dist, tgt_dist = lognorm(s=0.6, scale=np.exp(1.0)), lognorm(s=0.6, scale=np.exp(1.0) * 1.15)
    q_src = split_conformal_quantile(src_dist.rvs(5000, random_state=rng), alpha)
    ok_pac = ok_cor = ok_dir = 0; recal = 0
    for _ in range(draws):
        lab = tgt_dist.rvs(k, random_state=rng)
        ok_pac += tgt_dist.cdf(q_pac(lab, alpha, delta)) >= 1 - alpha
        q, br = certify_or_recalibrate(q_src, lab, alpha, delta)
        ok_cor += tgt_dist.cdf(q) >= 1 - alpha; recal += br == "recalibrated"
        ok_dir += tgt_dist.cdf(q_direct(lab, alpha)) >= 1 - alpha
    cert_nom = sum(certify_or_recalibrate(q_src, src_dist.rvs(2500, random_state=rng), alpha, delta)[1] == "certified"
                   for _ in range(300)) / 300
    q_slack = split_conformal_quantile(src_dist.rvs(5000, random_state=rng), alpha / 2)
    cert_slack = sum(certify_or_recalibrate(q_slack, src_dist.rvs(2500, random_state=rng), alpha, delta)[1] == "certified"
                     for _ in range(300)) / 300
    r_pac, r_cor, r_dir, r_rec = ok_pac / draws, ok_cor / draws, ok_dir / draws, recal / draws
    print(f"[B5 C-or-R] (a) PAC quantile   P(true cov >= .90) = {r_pac:.3f}  (expect >= {1-delta:.2f})")
    print(f"            (b) C-or-R, shifted P(true cov >= .90) = {r_cor:.3f}  (expect >= {1-delta:.2f}); recalibrated in {r_rec:.2f} of draws")
    print(f"            (c) no shift, k=2500: certified {cert_nom:.2f} with q_src at alpha (expect ~0), "
          f"{cert_slack:.2f} with q_src at alpha/2 (expect a majority)")
    print(f"            (d) direct SCP  P(true cov >= .90) = {r_dir:.3f}  (expect ~0.5-0.6: only marginally valid)")
    return bool(r_pac >= 1 - delta - 0.015 and r_cor >= 1 - delta - 0.015 and r_rec > 0.5
                and cert_nom < 0.1 and cert_slack > 0.5 and r_dir < 0.75)


def test_scene_ltt(rng):
    """
    H16 on synthetic CLUSTERED data (scenes with a shared random difficulty, ~60 agents each -- mimicking nuScenes):
      (a) agent-level C-or-R with whole-scene labels loses its guarantee (rate well below 1-delta), reproducing the
          real-data B10 finding;
      (b) scene-level LTT keeps it: P(scene-averaged coverage >= 1-alpha) >= 1-delta.
    Truth = scene-averaged coverage over a large fresh population of scenes (not the labelled ones).
    """
    from conformal import split_conformal_quantile
    from solutions import certify_or_recalibrate, scene_ltt_threshold
    alpha, delta = 0.10, 0.10

    def scenes(n_sc):
        diff = rng.lognormal(0.0, 0.35, n_sc) * 1.15                    # shifted target, scene-level difficulty
        return [rng.lognormal(1.0, 0.45, rng.integers(30, 90)) * d for d in diff]

    pop = scenes(3000)
    q_src = split_conformal_quantile(rng.lognormal(1.0, 0.6, 5000), alpha)
    ok_agent = ok_scene = inf = 0; draws = 150; m = 40
    for _ in range(draws):
        lab = scenes(m)
        qa, _ = certify_or_recalibrate(q_src, np.concatenate(lab), alpha, delta)
        qs, _ = scene_ltt_threshold(q_src, lab, alpha, delta)
        ok_agent += np.mean([(s <= qa).mean() for s in pop]) >= 1 - alpha
        ok_scene += np.mean([(s <= qs).mean() for s in pop]) >= 1 - alpha
        inf += np.isinf(qs)
    ra, rs = ok_agent / draws, ok_scene / draws
    print(f"[H16 scene-LTT] m={m} whole-scene labels: agent-level C-or-R rate={ra:.2f} (expect < {1-delta:.2f}); "
          f"scene-level LTT rate={rs:.2f} (expect >= {1-delta:.2f}); infinite thresholds {inf/draws:.2f}")
    return bool(rs >= 1 - delta - 0.03 and ra < rs)


def test_aci(rng):
    """ACI (Gibbs & Candes 2021): long-run miss rate -> alpha on a shifted stream despite a stale calibration set."""
    from solutions import aci_stream
    cal = rng.lognormal(1.0, 0.6, 5000)
    stream = rng.lognormal(1.0, 0.6, 20000) * 1.15
    out = {}
    for g in (0.005, 0.05):
        _, errs = aci_stream(cal, stream, 0.10, g)
        out[g] = errs.mean()
    print(f"[B6 ACI] long-run miss rate on shifted stream: gamma=.005 -> {out[0.005]:.4f}, gamma=.05 -> {out[0.05]:.4f} (expect ~0.100)")
    return all(abs(v - 0.10) < 0.01 for v in out.values())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    r = {"H4": test_h4(rng), "H5": test_h5(rng), "H6": test_h6(rng), "H5b": test_audit(rng),
         "B5": test_cor(rng), "B6": test_aci(rng), "H16": test_scene_ltt(rng)}
    print("\n" + "  ".join(f"{k}:{'OK' if v else 'FAIL'}" for k, v in r.items()))
    sys.exit(0 if all(r.values()) else 1)
