# Falsification Note — Paper 01

> Written BEFORE experiments. Committed. Do not edit the hypotheses or thresholds after
> results arrive — only fill in the outcome log.

**Date written:** 2026-09-10
**Git SHA at time of writing:** (see commit that adds this file)
**Author:** Hassan

---

## Setup being tested

- Model: AutoBot-Ego (~1.5M params), trained with UniTraj on one source dataset.
- Datasets (unified via ScenarioNet): Argoverse 2 (AV2), nuScenes (NS). Waymo (WO) / nuPlan
  (NP) added if feasible.
- Task: marginal motion forecasting, 2.1 s history → 6.0 s future @ 10 Hz, VEHICLE agents.
- Conformal method: **split conformal prediction (SCP)** on the ego future trajectory.
  - Nonconformity score baseline: `s(x,y) = min over the K predicted modes of the
    max-over-horizon L2 displacement error`, i.e. worst-timestep error of the best mode.
  - Prediction "region" at level alpha: the set of futures within radius `q_hat` (the
    ceil((n+1)(1-alpha))/n empirical quantile of calibration scores) of any predicted mode.
  - Coverage event: the ground-truth future lies within the region (all horizon steps of
    GT within `q_hat` of the best mode).
- "Nominal coverage" = `1 - alpha`. Tested alpha in {0.05, 0.10, 0.20}.
- Calibration set: held-out split of the SOURCE dataset, >= 2000 scenarios, never used for
  training or model selection.
- Seeds: 3 (model training seeds 0/1/2). Report mean +/- std.
- CIs: BCa bootstrap, 2000 resamples, clustered at scenario level.

---

## H1 — Coverage collapses under cross-dataset transfer

**Claim.** SCP calibrated on dataset A and applied to dataset B (B != A), with NO
recalibration and NO target labels, produces empirical coverage materially below nominal.

**Primary metric.** Coverage gap `Δ = nominal - empirical_coverage`, at alpha = 0.10
(nominal 0.90), averaged over all ordered A→B pairs with A != B.

**Prediction.** Mean `Δ_cross` >= 0.05 (i.e. >= 5 coverage points lost), and for at least
one pair `Δ >= 0.10`. In-distribution control `Δ_in` (A→A) within +/- 0.02.

**REFUTED IF:** mean `Δ_cross < 0.02` across all pairs at alpha=0.10 (coverage essentially
holds), OR `Δ_in` itself exceeds 0.03 (meaning the SCP implementation is broken and the
cross-dataset number is uninterpretable — this is a bug, not a refutation, fix and rerun).

**Note.** A confirmed H1 is the paper even if H2/H3 fail. "The distribution-free guarantee
is not distribution-free across datasets" is the contribution.

---

## H2 — The gap is attributable to measurable shift factors

**Claim.** Most of the cross-dataset coverage gap is explained by a small set of
pre-registered, measurable scene-distribution differences, not an irreducible domain gestalt.

**Pre-registered shift factors** (computed per scene, then aggregated per dataset; see
`src/shift_factors.py`):
1. effective sampling rate / timestep dt
2. mean # tracked agents within 50 m of ego
3. map element density (polyline points per 100 m^2 in the local crop)
4. lane-graph branching factor at ego position
5. ego speed distribution (mean, p90)
6. fraction of scenes with ego turning (|heading change| > 30 deg over horizon)
7. curvature of the GT future (mean |dtheta/ds|)

**Metric.** Fit a regression of per-scene nonconformity score (or per-bin coverage gap) on
the 7 factors, pooled across datasets. Report adjusted R^2 and, via a
calibrate-on-A-reweighted-to-B analysis, the fraction of `Δ_cross` removed when calibration
scores are importance-weighted to match B's factor distribution.

**Prediction.** The 7 factors explain adjusted R^2 >= 0.35 of score variance, AND
factor-reweighting the calibration set removes >= 40% of `Δ_cross` on average.

**REFUTED IF:** adjusted R^2 < 0.20 OR reweighting removes < 25% of `Δ_cross`. Then the gap
is mostly not captured by these observables — report that honestly; it weakens but does not
kill the paper (H1 + H3 still stand).

---

## H3 — Label-free recalibration restores coverage

**Claim.** A recalibration procedure that uses only UNLABELLED target-domain scenes (no
target ground-truth futures) restores empirical coverage close to nominal.

**Candidate methods** (decide by ablation, pre-registered as a set):
- (a) Importance-weighted SCP: density-ratio `w(x) = p_B(x)/p_A(x)` estimated from the 7
  shift factors (logistic domain classifier), applied to calibration scores.
- (b) Normalised nonconformity: divide the score by a learned scale `sigma(x)` predicted
  from scene features, fit on source, so the score is scene-adaptive before quantiling.
- (c) Group-conditional SCP: bin by a dominant factor (e.g. agent density tertile), quantile
  within bin, route target scenes to their bin.

**Metric.** Post-recalibration coverage gap `Δ_recal` at alpha=0.10, mean over A→B pairs.
Also report mean region size (efficiency) vs uncorrected.

**Prediction.** Best method achieves `Δ_recal <= 0.03` (within 3 points of nominal) while
inflating mean region size by <= 25% vs uncorrected SCP.

**REFUTED IF:** no method gets `Δ_recal <= 0.05`, OR the only method that does inflates
region size > 50% (achieving coverage by making the region uselessly large — not a fix).

---

## Kill conditions for the whole paper

- If `Δ_in` cannot be driven below 0.03 after debugging → SCP implementation is wrong;
  no result is interpretable. Fix before proceeding.
- If AutoBot cannot be trained to within ~15% of the UniTraj-reported AV2 minADE on this
  hardware → the base model is not competent enough for the coverage claim to matter;
  reassess (try a stronger frozen checkpoint, or Kaggle for a bigger model).

---

## Outcome log (fill AFTER experiments — do not touch above)

| Hypothesis | Metric value | Threshold | Supported? | Notes |
|---|---|---|---|---|
| H1 (Δ_in)        |  | <= 0.03 |  |  |
| H1 (Δ_cross)     |  | >= 0.05 mean |  |  |
| H2 (adj R^2)     |  | >= 0.35 |  |  |
| H2 (reweight removes) |  | >= 40% |  |  |
| H3 (Δ_recal)     |  | <= 0.03 |  |  |
| H3 (region infl) |  | <= 25% |  |  |

**Rule:** a refuted hypothesis gets reported as refuted. A clean negative on H2 or H3 is
still publishable if H1 holds.
