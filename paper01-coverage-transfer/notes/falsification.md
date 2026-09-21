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

### Run 1 — 2026-09-21 — source `av2_cpu_v1` (AutoBot, CPU-trained) -> target nuScenes val
Reproduce: `python src/analyze_pair.py --src av2_cpu_v1 --cal av2cal --same av2test --target ns`
(full numbers: `results/pair_av2_cpu_v1__ns.json`). alpha=0.10, calibration = random 50% of the
AV2 held-out calibration split, 10 seeds. **One source->target pair only. Weak base model (see below).**

| Hypothesis | Metric value | Threshold | Supported? | Notes |
|---|---|---|---|---|
| H1 (D_in) | +0.005 (same-domain held-out: -0.002) | <= 0.03 | **Yes** | SCP implementation is correct on real predictions. |
| H1 (D_cross) | **+0.034** (95% CI +0.027..+0.041; seed range +0.026..+0.042) | >= 0.05 mean | **No (magnitude)** | A real, statistically unambiguous under-coverage (86.6% vs 90% nominal, n=9041), in the unsafe direction, at all 3 alphas (+0.023 / +0.034 / +0.031). But BELOW the pre-registered >=0.05, and only one pair exists so the "at least one pair >=0.10" clause cannot be met. Refutation bar (<0.02) NOT hit. |
| H2 (adj R^2) | **0.147** (within-AV2 0.121, within-nuScenes 0.170) | >= 0.35 (refute < 0.20) | **REFUTED** | The 7 pre-registered factors explain little of score variance. |
| H2 (reweight removes) | **-57%** (range -93%..-28%) | >= 40% (refute < 25%) | **REFUTED** | Importance weighting made coverage WORSE (84.7% vs 86.6%). |
| H3 (D_recal) | best label-free method (group) +0.036; uncorrected +0.034 | <= 0.03 | **No** | No method beats doing nothing. Weighted +0.053, normalized +0.086, group +0.036. Literal refutation clause ("no method <= 0.05") is NOT triggered only because the uncorrected baseline (0.034) already sits under 0.05 -- the threshold was set too loosely; the substantive claim (a repair exists) is unsupported. |
| H3 (region infl) | weighted 0.93x, normalized 0.88x, group 0.99x of uncorrected radius | <= 25% | n/a | Methods did not achieve coverage, so efficiency is moot. Note weighting/normalizing SHRANK the region while coverage was already short. |

**Interpretation (hypotheses, not established):** weighting on observable scene covariates moved
the calibration quantile the wrong way (radius 5.91 -> 5.50): the factors that distinguish nuScenes
from AV2 (fewer lanes, lower speed, fewer turns) are ones associated with LOWER error in AV2, yet
nuScenes errors are HIGHER (mean score 3.96 vs 3.42). That pattern is what conditional (label/concept)
shift looks like, which covariate reweighting cannot fix by construction. Untested alternatives:
(a) 2 Hz-native nuScenes ground truth interpolated to 10 Hz vs native 10 Hz AV2 (gt_future_curvature:
AV2 5.66 vs nuScenes 0.08; native_dt perfectly collinear with dataset) -- TESTED 2026-09-21 by scoring only
at native 2 Hz / 1 Hz steps: gap +3.5 / +3.6 pts vs +3.4 at 10 Hz, so GT interpolation does NOT explain the
gap (results/confound_check_sampling_rate.json); the interpolated model INPUT history is still untested; (b) different tracking/annotation noise; (c) map-frame differences.

**Known limits of Run 1:** single source->target pair; single model, single training seed; base model is
weak (AV2 held-out miss rate ~52%; zero-shot nuScenes minADE5 = 1.99 vs published nuScenes-trained AutoBot
1.26-1.37 -- not apples-to-apples since ours is zero-shot and under-trained: 10 epochs, batch 32, LR never
decayed, versus UniTraj's 8xA100 / batch 128); no reverse direction (nuScenes -> AV2).

**Rule:** a refuted hypothesis gets reported as refuted. A clean negative on H2 or H3 is
still publishable if H1 holds.

---

## Errata (2026-09-21) -- appended, original text above left intact

The kill condition above cites "UniTraj-reported AV2 minADE ... (~0.73)". That figure was written from
memory and **could not be verified** against the UniTraj paper (papers/UniTraj_ECCV2024.pdf). The only
AutoBot reference numbers verifiable there are on **nuScenes**, minADE5: AutoBot-UniTraj 1.26, vanilla
AutoBot 1.37 (trained on 8xA100, batch 128, ~5 min/epoch). The "0.85 / +15%" threshold derived from the
unverified figure should be treated as unfounded. The base-model gate must be re-defined against a
verified reference (e.g. nuScenes minADE5 with the challenge protocol) before it is used to accept or
reject a model.
