"""
Solution-side methods for Paper 01 (definitions fixed in notes/falsification.md, Addendum A).
Pure numpy/sklearn, model-agnostic, no UniTraj dependency.

  H4  normalized_scores / region_area_proxy      model-uncertainty-normalised nonconformity (label-free)
  H5  label_budget                                few-label recalibration: direct / pooled / shrinkage(k0)
  H6  output_features / domain_auc                label-free shift monitor
"""
from __future__ import annotations

import numpy as np

from conformal import split_conformal_quantile

# ---------------------------------------------------------------------------------------------
# H4 -- uncertainty-normalised score
# ---------------------------------------------------------------------------------------------


def _mode_worst(pred_trajs: np.ndarray, gt: np.ndarray, gt_mask: np.ndarray | None) -> np.ndarray:
    """(N, K): max over valid horizon steps of ||pred_k - gt||."""
    err = np.linalg.norm(pred_trajs - gt[:, None, :, :], axis=-1)          # (N, K, T)
    if gt_mask is not None:
        err = np.where(gt_mask[:, None, :], err, -np.inf)
    return err.max(axis=-1)


def normalized_scores(pred_trajs, gt, gt_mask, pred_scale) -> np.ndarray:
    """
    s'_i = min_k  max_t ||e_{k,t}|| / u_{i,k},   u_{i,k} = mean_t sqrt(bx^2+by^2)   (predicted Laplace scale).
    Coverage event  s'_i <= q'  <=>  the truth lies in the union over modes of balls of radius q' * u_{i,k}.
    """
    return (_mode_worst(pred_trajs, gt, gt_mask) / np.maximum(pred_scale, 1e-6)).min(axis=1)


def region_area_proxy(q: float, n: int, k: int, pred_scale: np.ndarray | None = None) -> np.ndarray:
    """
    Per-sample area proxy of the union region: sum_k pi r_k^2 (over-counts overlaps, but identical for every
    method so ratios are meaningful).  Raw score: r_k = q for all modes.  Normalised: r_k = q * u_k.
    """
    if pred_scale is None:
        return np.full(n, k * np.pi * q ** 2)
    return (np.pi * (q * pred_scale) ** 2).sum(axis=1)


# ---------------------------------------------------------------------------------------------
# H5 -- few-label recalibration
# ---------------------------------------------------------------------------------------------


def q_direct(tgt_scores: np.ndarray, alpha: float) -> float:
    """Exact split-conformal on the k labelled target scores only (finite-sample valid)."""
    return split_conformal_quantile(np.asarray(tgt_scores, float), alpha)


def q_pooled(src_scores: np.ndarray, tgt_scores: np.ndarray, alpha: float) -> float:
    """Source calibration scores UNION k target scores, unweighted."""
    return split_conformal_quantile(np.concatenate([src_scores, tgt_scores]), alpha)


def q_shrink(q_src: float, q_tgt: float, k: int, k0: float) -> float:
    """q = (1-w) q_src + w q_tgt,  w = k/(k+k0)."""
    w = k / (k + k0)
    return (1 - w) * q_src + w * q_tgt


ESTIMATORS_DEFAULT_K0 = (100, 500, 2000)


def label_budget(src_cal_scores, tgt_scores, ks=(25, 50, 100, 250, 500, 1000, 2500), alpha=0.10,
                 draws=200, k0s=ESTIMATORS_DEFAULT_K0, seed=0, tol=0.02) -> dict:
    """
    For each k: draw k labelled target scenes at random (`draws` times), fit each estimator, evaluate coverage
    on the REMAINING target scenes (disjoint).  Returns per-estimator arrays of coverage error
    (coverage - nominal) plus summaries.  Estimators: source_only, direct, pooled, shrink_k0=<k0>.
    """
    src = np.asarray(src_cal_scores, float)
    tgt = np.asarray(tgt_scores, float)
    n = len(tgt)
    nominal = 1 - alpha
    q_src = split_conformal_quantile(src, alpha)
    rng = np.random.default_rng(seed)
    names = ["source_only", "direct", "pooled"] + [f"shrink_k0={k0}" for k0 in k0s]
    out = {"alpha": alpha, "nominal": nominal, "n_target": n, "draws": draws, "tol": tol,
           "q_src": float(q_src), "ks": list(ks), "estimators": {}}
    for nm in names:
        out["estimators"][nm] = {}
    ks = [k for k in ks if k <= n - 500]         # keep >= 500 evaluation scenes per draw
    out["ks"] = list(ks)
    for k in ks:
        errs = {nm: np.empty(draws) for nm in names}
        for d in range(draws):
            perm = rng.permutation(n)
            lab, ev = tgt[perm[:k]], tgt[perm[k:]]
            qd = q_direct(lab, alpha)
            qs = {"source_only": q_src, "direct": qd, "pooled": q_pooled(src, lab, alpha)}
            for k0 in k0s:
                qs[f"shrink_k0={k0}"] = q_shrink(q_src, qd, k, k0)
            for nm in names:
                errs[nm][d] = float((ev <= qs[nm]).mean()) - nominal
        for nm in names:
            e = errs[nm]
            out["estimators"][nm][str(k)] = {
                "mean_signed_err": float(e.mean()), "mean_abs_err": float(np.abs(e).mean()),
                "sd": float(e.std()), "p_within_tol": float((np.abs(e) <= tol).mean()),
                "p5": float(np.percentile(e, 5)), "p95": float(np.percentile(e, 95)),
            }
    out["k_star"] = k_star(out)
    return out


def k_star(res: dict, prob: float = 0.90) -> dict:
    """Smallest k with P(|err| <= tol) >= prob, per estimator (None if never reached in the grid)."""
    out = {}
    for nm, by_k in res["estimators"].items():
        hit = [int(k) for k, v in by_k.items() if v["p_within_tol"] >= prob]
        out[nm] = min(hit) if hit else None
    return out


# ---------------------------------------------------------------------------------------------
# H6 -- label-free shift monitor
# ---------------------------------------------------------------------------------------------


def output_features(pred_probs: np.ndarray, pred_scale: np.ndarray, pred_trajs: np.ndarray) -> np.ndarray:
    """Label-free model-output features: top-mode predicted scale, mode-probability entropy, mode dispersion."""
    top = pred_probs.argmax(1)
    u_top = pred_scale[np.arange(len(top)), top]
    p = np.clip(pred_probs, 1e-9, 1.0)
    ent = -(p * np.log(p)).sum(1)
    fin = pred_trajs[:, :, -1, :]                                         # (N, K, 2) final positions
    mu = (pred_probs[..., None] * fin).sum(1, keepdims=True)
    disp = np.sqrt((pred_probs * ((fin - mu) ** 2).sum(-1)).sum(1))
    return np.column_stack([np.log(u_top), ent, disp]).astype(np.float32)


def domain_auc(a: np.ndarray, b: np.ndarray, folds: int = 5, seed: int = 0, max_n: int = 4000) -> float:
    """Cross-validated AUC of a logistic domain classifier separating sample sets a and b (label-free)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    if len(a) > max_n:
        a = a[rng.choice(len(a), max_n, replace=False)]
    if len(b) > max_n:
        b = b[rng.choice(len(b), max_n, replace=False)]
    X = np.vstack([a, b]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    aucs = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------------------------
# H5b -- label-efficient coverage AUDIT (does the source calibration still hold on the target?)
# ---------------------------------------------------------------------------------------------


def pac_order_index(k: int, alpha: float, delta: float) -> int | None:
    """
    Training-conditional (PAC) split conformal (Vovk 2012): with k exchangeable calibration scores, the coverage of
    the l-th order statistic is Beta(l, k+1-l), so P(coverage >= 1-alpha) = P(Bin(k, 1-alpha) <= l-1).
    Returns the smallest l with that probability >= 1-delta, or None if even l = k is not enough (k too small).
    """
    from scipy.stats import binom
    ls = np.arange(1, k + 1)
    ok = binom.cdf(ls - 1, k, 1 - alpha) >= 1 - delta
    return int(ls[ok][0]) if ok.any() else None


def q_pac(tgt_scores, alpha: float, delta: float) -> float:
    """Threshold with coverage >= 1-alpha w.p. >= 1-delta over the draw of the k labelled scores (inf if k too small)."""
    s = np.sort(np.asarray(tgt_scores, float))
    l = pac_order_index(len(s), alpha, delta)
    return float(s[l - 1]) if l is not None else float("inf")


def cp_upper(m: int, k: int, level: float) -> float:
    """One-sided Clopper-Pearson upper confidence bound (confidence 1-level) on a miss probability, m misses of k."""
    from scipy.stats import beta
    return 1.0 if m >= k else float(beta.ppf(1 - level, m + 1, k - m))


def certify_or_recalibrate(q_src: float, lab_scores, alpha: float, delta: float) -> tuple[float, str]:
    """
    Addendum B5.  (i) keep q_src iff the Clopper-Pearson upper bound (level delta/2) on its target miscoverage is
    <= alpha; (ii) otherwise return the PAC target quantile at level delta/2.  Union bound: the returned threshold
    covers >= 1-alpha with probability >= 1-delta over the label draw, whichever branch fires.
    """
    lab = np.asarray(lab_scores, float)
    m = int((lab > q_src).sum())
    if cp_upper(m, len(lab), delta / 2) <= alpha:
        return float(q_src), "certified"
    return q_pac(lab, alpha, delta / 2), "recalibrated"


def aci_stream(cal_scores, stream_scores, alpha: float, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptive conformal inference (Gibbs & Candes 2021) on a label stream, fixed (source) calibration set:
      q_t = Quantile_{1-alpha_t}(cal),  err_t = 1{s_t > q_t},  alpha_{t+1} = alpha_t + gamma (alpha - err_t).
    alpha_t <= 0 -> q_t = +inf (always cover); alpha_t >= 1 -> q_t = -inf.  Returns (thresholds q_1..q_{T+1}, errs).
    """
    cal = np.sort(np.asarray(cal_scores, float))
    n = len(cal)
    at = alpha
    qs, errs = np.empty(len(stream_scores) + 1), np.empty(len(stream_scores))

    def q_of(a):
        if a <= 0:
            return np.inf
        if a >= 1:
            return -np.inf
        r = int(np.ceil((n + 1) * (1 - a)))
        return np.inf if r > n else cal[r - 1]

    for t, s in enumerate(stream_scores):
        qs[t] = q_of(at)
        errs[t] = float(s > qs[t])
        at = at + gamma * (alpha - errs[t])
    qs[-1] = q_of(at)
    return qs, errs


def audit_power(q_src: float, tgt_scores, ks=(25, 50, 100, 250, 500, 1000, 2500), alpha=0.10,
                draws=500, level=0.05, seed=0) -> dict:
    """
    Audit rule: with k labelled target scenes, count misses M = #(score > q_src) and reject 'calibration still valid'
    if the one-sided binomial test  P(Bin(k, alpha) >= M) <= level  (i.e. evidence of UNDER-coverage).
    Returns P(reject) per k on the given target scores (power if the target is shifted, false-alarm rate if it is
    not).  Exact binomial tail, no distributional assumption beyond exchangeability of the k draws.
    """
    from scipy.stats import binom
    tgt = np.asarray(tgt_scores, float)
    rng = np.random.default_rng(seed)
    out = {"alpha": alpha, "level": level, "q_src": float(q_src), "true_gap": float(alpha - (tgt > q_src).mean()) * -1.0,
           "reject_prob": {}}
    for k in [k for k in ks if k < len(tgt)]:      # cannot draw more labelled scenes than exist
        rej = 0
        for _ in range(draws):
            m = int((tgt[rng.choice(len(tgt), k, replace=False)] > q_src).sum())
            rej += binom.sf(m - 1, k, alpha) <= level
        out["reject_prob"][str(k)] = rej / draws
    return out
