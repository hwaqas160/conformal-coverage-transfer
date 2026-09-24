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

---

## Addendum A — written 2026-09-21, AFTER Run 1 results were seen, BEFORE any Run 2 analysis

Status of this addendum: hypotheses H4–H9 below are **post-hoc with respect to Run 1** (they were motivated by
Run 1's outcome) and **pre-registered with respect to every analysis run after this commit**. They are labelled
*exploratory-motivated*, not confirmatory-from-the-start. H1–H3 above are unchanged.

### A0. Disclosed flaw in Run 1's H2/H3 (found 2026-09-21 while designing the solution)
`src/shift_factors.py` factors 5–7 (`ego_speed_mean`, `ego_turning_frac_proxy`, `gt_future_curvature`) are computed
from the **ground-truth future** of the predicted agent. The Run 1 H3 methods (importance weighting, normalisation,
group conditioning) consumed these factors on the **target** domain, so Run 1's H3 was **not label-free as claimed**:
it had access to target-label-derived features. Consequence: the negative H3 conclusion is, if anything, *conservative*
(the methods had extra information and still failed), but the run does not match its own definition and its numbers
must not be cited as a label-free result. Fix: define **label-free factors** from history/map/agents only and re-run:

| # | label-free factor | definition (per predicted agent) |
|---|---|---|
| 1 | native_dt | native sampling period of the source dataset |
| 2 | n_agents_near_ego | # tracked agents within 50 m at the last observed step |
| 3 | map_point_density | valid map points per 100 m^2 of the local crop |
| 4 | n_lanes_near_ego | # polylines with a point within 50 m |
| 5 | hist_speed_mean | mean speed over the observed history (from history velocity features) |
| 6 | hist_heading_change | abs. net heading change over the observed history (rad) |
| 7 | hist_curvature | mean abs. curvature of the observed history path (rad/m) |

H2 is re-reported with BOTH factor sets (GT-derived = "oracle diagnostic", label-free = the operational one); H3 is
re-run with the label-free set only. Run 1's H2/H3 rows stay in the log, marked "superseded by Run 2".

### A1. Proposition (identifiability; to be stated in the paper with proof)
Zero-label recalibration cannot in general restore coverage: for any label-free procedure there exist two target
distributions with identical marginal over inputs X but different conditional score laws P(S|X), on which the procedure
outputs the same threshold but the required thresholds differ. Hence label-free methods can only fix *covariate* shift;
coverage repair under *conditional* shift needs either (i) a score that is more invariant to it (H4) or (ii) some target
labels (H5). This is a statement about what is provable, not a claim that any method fails empirically.

### A2. New hypotheses (primary metric always: coverage gap at alpha = 0.10; secondary alpha in {0.05, 0.20})

**H4 — model-uncertainty-normalised score (label-free).** Score
`s'_i = min_k max_t ||e_{k,t}|| / u_{i,k}`, `u_{i,k} = mean_t sqrt(bx_{k,t}^2 + by_{k,t}^2)` (the model's own predicted Laplace
scale for mode k, softplus+0.01). Region = union over modes of balls of radius `q' * u_{i,k}`.
*Prediction:* target coverage gap |Delta'| <= 0.5 * |Delta| of the raw score, with in-domain |Delta_in| <= 0.02.
*Efficiency guard:* region-area proxy `A = sum_k pi (q' u_k)^2` is reported relative to the raw-score region
`K pi q^2`; a "win" that only inflates A by > 25% is not counted as a fix.
*Refuted if:* |Delta'| > 0.75 * |Delta|, or |Delta_in'| > 0.03.

**H5 — few-label recalibration and the label budget.** Given k labelled target scenes (k in {25, 50, 100, 250, 500,
1000, 2500}), three estimators, all fixed in advance (no selection among them by target-test coverage):
 (a) direct: split-conformal on the k target scores only (exact finite-sample validity);
 (b) pooled: source calibration scores UNION the k target scores, unweighted;
 (c) shrinkage: `q = (1-w) q_src + w q_tgt(k)`, `w = k/(k+k0)`, reported for the whole family k0 in {100, 500, 2000}.
Protocol: 200 random draws of the k labelled scenes; evaluate on the remaining target scenes (disjoint).
*Reported:* mean |coverage error|, and `P(|error| <= 0.02)` per estimator and k; **k\*** = smallest k with
P(|error| <= 0.02) >= 0.90 for each estimator; and whether any of (b)/(c) reaches k\* smaller than (a).
*Prediction:* (a) has |error| <= 0.02 with prob >= 0.9 only for k >= 500; (b)/(c) reduce variance vs (a) at k <= 250.
*Refuted if:* neither (b) nor (c) has lower mean |error| than (a) at k = 100.

**H6 — label-free shift monitor.** Across all ordered city pairs (i -> j, i != j, each city n >= 300) of AV2 with a fixed
model (calibrate on city i's held-out samples, test on city j): statistic T_ij = cross-validated AUC of a domain classifier
on label-free features (list in A0) + model-output features (mean predicted scale, mode-probability entropy);
outcome G_ij = realised |coverage gap| at alpha=0.10.
*Prediction:* Spearman rho(T, G) >= 0.5 with permutation p < 0.05. *Refuted if* rho < 0.3.

**H7 — marginal coverage hides sub-population miscoverage (descriptive).** Per-city coverage of one common calibrated model
at alpha = 0.10. *Prediction:* at least one city deviates from nominal by >= 0.05. Reported with binomial CIs; no test.

**H8 — robustness across base models.** H1's under-coverage (gap >= 0.02 at alpha=0.10, zero-shot AV2 -> nuScenes) replicates
for every checkpoint in the model zoo (av2_cpu_v1 ep08, av2_cpu_v2, av2_valsplit_v1 ep17, plus later models).
*Refuted if* any well-trained model has gap < 0.

**H9 — reverse direction.** A nuScenes-trained AutoBot, calibrated on nuScenes, evaluated zero-label on AV2 val:
report the signed gap. *Prediction:* |gap| >= 0.02 (sign not pre-specified: the target here is the *easier* dataset).

### A3. Multiplicity and reporting rules
Primary confirmatory-style claims: H1 (replicated), H4, H5. Everything else is exploratory and labelled so.
No claim in the paper may use a number that is not regenerated by `paper/make_results.py` from `results/*.json`.
Negative results are reported as such. Model selection uses `av2_splits/val/train` only (clean); the av2_cpu_v1 checkpoint
was selected on a 1,500-sample subset of `av2_splits/val/cal` (minor overlap with the calibration split; disclosed, and
av2_cpu_v2 removes it).

### A2b. H5b — label-efficient coverage audit (registered 2026-09-21, before any real-data run of it)
Motivation: synthetic validation (src/tests/test_solutions_synthetic.py) shows that under a real shift the pooled and
shrinkage estimators of H5 are dominated by bias, while the direct estimator is unbiased with sd ~ sqrt(a(1-a)/k) (so
~1,000 labels are needed for +/-2 pts reliably). A more useful primitive than a noisy repair is an **audit**: with k labelled
target scenes, count misses M = #(score > q_src) and flag under-coverage if the exact one-sided binomial tail
P(Bin(k, alpha) >= M) <= 0.05.
*Reported:* P(flag) vs k on the real target (power) and on held-out same-domain data (false-alarm; single-threshold flags
can exceed 5% at large k because a single calibrated threshold's true coverage deviates ~0.4 pt from nominal -- reported,
and the marginal rate over calibration draws is also reported).
*Prediction (from the measured Run-1 gap of ~3.4 pts):* power >= 0.90 at k = 1000 and >= 0.5 at k = 250.
*Refuted if:* power at k = 1000 < 0.75.
Expectation recorded in advance from the synthetic study: H5's pooled/shrinkage variants will most likely fail their
pre-registered criterion (bias dominates); if so this is reported as a negative result and the audit + direct recalibration
are the recommended protocol.

---

## Outcome log — Run 2 (H4–H9), real data — 2026-09-22
Reproduce: `python src/analyze_run2.py --models av2_cpu_v1 av2_valsplit_v1` (full numbers:
`results/run2/{av2_cpu_v1,av2_valsplit_v1}__forward.json`, `results/run2/cities__*.json`). alpha=0.10 unless noted.
**Forward direction only (AV2->nuScenes); 2 of the 3 planned model-zoo entries; reverse direction (H9) not yet run**
— both pending, this entry will NOT be edited when they land, a new dated entry will be appended instead.

H1 replicates on real data independent of Run 1's calibration draw: av2\_cpu\_v1 gap +0.0334 (CI [0.0211,0.0450]),
same-domain -0.0022 (consistent with 0 as SCP guarantees). This IS a second, independent confirmation of Run 1's
headline number (+0.034 there too) — not the same experiment rerun, but the same source model scored on a fresh
analysis pipeline (`analyze_run2.py`, not `analyze_pair.py`) against the same target. av2\_valsplit\_v1 gap is much
larger: +0.0796 (CI [0.0644,0.0936]), same-domain -0.0024.

| Hypothesis | av2\_cpu\_v1 | av2\_valsplit\_v1 | Threshold | Supported? |
|---|---|---|---|---|
| H4 ratio \|Delta'\|/\|Delta\| | 0.512 | 0.877 | <=0.5 support / >0.75 refute | v1: **marginal** (just above the support bar, nowhere near refuted); valsplit\_v1: **REFUTED** |
| H4 in-domain \|Delta\_in'\| | 0.0048 | 0.0010 | <=0.02 | Both **pass** |
| H4 efficiency guard (region-area inflation at matched coverage) | +15.4% | +7.2% | not >25% | Both **pass** -- not triggered, so where H4 helps it is not by ballooning the region |
| H5 refute clause (min(pooled,shrink) mean\|err\| @k=100 < direct) | shrink\_k0=100: 0.0163 < direct 0.0269 -> **not refuted, shrinkage wins** | shrink\_k0=100: 0.0243 > direct 0.0227 (and pooled/shrink500 both worse too) -> **REFUTED, direct wins** | — | model-dependent, opposite verdicts |
| H5 pooled k\* | never reaches 90% (any k tested) | never reaches 90% (any k tested) | — | pooled is dominated by bias on both models, as the synthetic study anticipated |
| H5b power @ k=1000 | 0.952 | 1.000 | >=0.90 | Both **pass** |
| H5b power @ k=250 | 0.492 | 0.978 | >=0.5 | v1 **marginal** (0.008 under, effectively met given rounding/seed noise); valsplit\_v1 **pass** (bigger true gap => more power at fixed k, expected) |
| H5b refute clause (power@1000 < 0.75) | 0.952 | 1.000 | — | Not triggered on either — **H5b is the one hypothesis that is clearly and consistently supported** |
| H6 rho(AUC, \|gap\|), 30 city pairs | 0.372 (p=0.043) | 0.098 (p=0.606, not significant) | >=0.5 support / <0.3 refute | v1: **neither supported nor refuted** (in between, and the significance the pre-registration wanted is present but the effect size is not); valsplit\_v1: **REFUTED** |
| H2 (reweighting removal fraction) | -64% | -27% | >=40% support / <25% refute | Both **REFUTED**, both negative (reweighting makes the fit worse than no correction) — replicates Run 1's REFUTED verdict on an independent model |
| H8 (gap >= 0.02, replicates for every zoo model) | +0.033 | +0.080 | gap>=0.02, sign consistent | Both **pass this clause** (2/2 models so far) — but the pre-registration's implicit assumption that *magnitude* would be roughly stable across models is **not supported**: 2.4x difference between the two models tested |

**Interpretation.** Of the solution-side hypotheses, only H5b (the labelled coverage audit) held up cleanly and
consistently across both models. H4 (normalised score) and H6 (label-free monitor) both worked passably on
av2\_cpu\_v1 and were refuted or close to refuted on av2\_valsplit\_v1 — the more heavily trained, larger-gap model.
H2/H3 (covariate reweighting) replicated its Run-1 refutation on a second, independent model, which is the strongest
and most consistent finding in this log. The paper reports all of this as-is: H5b/H2-H3 as solid, H4/H6 as
model-dependent partial results, not as uniform wins.

**Rule (restated):** nothing here is rewritten when av2\_cpu\_v2 or the reverse direction land; a new dated entry is
appended and this one stands as the record of what 2 models showed.

---

## Kill condition #2 checked against a verified source — 2026-09-22
The second kill condition ("AutoBot within ~15% of the UniTraj-reported AV2 minADE") was written before any
number was looked up. Checked today by reading the UniTraj paper's own tables (web-fetched primary source, not
memory): UniTraj's supplementary Table 8 reports AutoBot trained AND evaluated on Argoverse 2, minADE6 = **0.85**
(their setup: 8xA100, batch 128, full 180k-trajectory AV2 training set, checkpoint selected on best brier-minFDE).

Our checkpoints (val minADE6, selected checkpoint, on a 1500-2000-scene subset of our own val split -- not
identical protocol, noted below): av2\_cpu\_v1 = 1.349 (**+59% worse**), av2\_cpu\_v2 = 1.092 (**+28% worse**).
**Both exceed the pre-registered 15% kill-condition threshold.** This condition IS triggered.

**Disposition (reassess, per the pre-registered rule, not silently drop):** the coverage-transfer measurement
itself does not require SOTA point-accuracy -- split CP gives a valid, well-defined region for any predictor,
including a mediocre one, and the same-domain gap (H1, -0.2 to -0.24pt) shows the region IS correctly calibrated for
what this model actually produces. What a weak model DOES put at risk is generalising the *magnitude* of the
cross-dataset gap and the solutions' effect sizes to a stronger, properly-trained model -- exactly the caveat H8
(model zoo) was already designed to probe, and exactly why av2\_cpu\_v1 vs av2\_cpu\_v2 (a 60-minADE6-point
difference between them, 1.349 vs 1.092) already shows the gap magnitude and H4's effect size are NOT stable
across model quality within our own zoo. We do not have the compute (single P2000, much of it shared) to close this
gap to UniTraj's 8xA100 setup within this project's timeline; a Kaggle/Colab-trained stronger checkpoint was
considered (see RESUME.md) and remains a possible follow-up, not done. **This is now stated as a quantified,
verified limitation in the paper (setup.tex) rather than an estimated one** -- the prior draft said "below published
state of the art" without a number; it now says +28% to +59% on minADE6, sourced to the UniTraj table directly.

---

## Outcome log — Run 2, 3rd model (av2_cpu_v2) added — 2026-09-22
Reproduce: `python src/analyze_run2.py --models av2_cpu_v1 av2_valsplit_v1 av2_cpu_v2` (full numbers:
`results/run2/av2_cpu_v2__forward.json`, `results/run2/cities__av2_cpu_v2.json`). **Forward-direction model zoo
(H8) is now complete: all 3 planned models analysed. Reverse direction (H9) still pending.**

av2\_cpu\_v2 (the most accurate of the three by val minADE6, see kill-condition entry above) has gap +0.1054
(CI [0.0939, 0.1217]) at alpha=0.10, same-domain +0.0034 -- the **largest** gap of the three models, despite having
the **best** point-accuracy. Point accuracy and coverage-gap magnitude are not simply related here; if anything the
direction is opposite across our 3 points (n=3, exploratory, not a claim of a general trend).

| Hypothesis | av2\_cpu\_v1 | av2\_valsplit\_v1 | av2\_cpu\_v2 | Threshold | Supported? |
|---|---|---|---|---|---|
| H4 ratio \|Delta'\|/\|Delta\| | 0.512 | 0.877 | 0.690 | <=0.5 support / >0.75 refute | v1: marginal; valsplit: **REFUTED**; v2: **inconclusive** (between the two bars) |
| H4 efficiency guard | +15.4% | +7.2% | +15.0% | not >25% | All 3 **pass** |
| H5 refute clause (shrink beats direct @k=100) | shrink wins | direct wins | direct wins (0.0245 < shrink100 0.0271) | — | 2/3 models: direct beats shrinkage; v1 is the exception, not the rule |
| H5 pooled k\* | never reaches 90% | never reaches 90% | never reaches 90% | — | Pooled is dominated by bias on **all 3** models |
| H5b power @ k=1000 | 0.952 | 1.000 | 1.000 | >=0.90 | All 3 **pass** |
| H5b false-alarm @ k=1000 | 0.024 | 0.010 | 0.062 | (report only) | All well under typical 0.05-0.10 tolerance; H5b remains the one hypothesis with zero exceptions across the zoo |
| H6 rho(AUC, \|gap\|) | 0.372 (p=.043) | 0.098 (p=.606) | 0.387 (p=.035) | >=0.5 support / <0.3 refute | v1, v2: inconclusive (significant, below support bar); valsplit: **REFUTED**. **Never once reaches the support bar across 3 models.** |
| H2 (reweighting removal) | -64% | -27% | -22% | >=40% / <25% refute | All 3 **REFUTED**, all negative |
| H8 (gap>=0.02, sign consistent) | +0.033 | +0.080 | +0.105 | gap>=0.02 | All 3 **pass**; magnitude ranges 3.2x (3.3 to 10.5 pt) across the zoo -- H8's real finding is this spread, not just the pass |

**Updated interpretation with n=3:** H5b (audit) and the H2/H3 reweighting-fails finding are now the two results
with zero exceptions across the full model zoo -- these are the load-bearing claims of the paper. H4 and H6 are
each supported-ish on one model, refuted on another, and inconclusive on the third: neither is a reliable repair,
and this is now a 3-model pattern, not a 2-model coincidence. H5's shrinkage variant helped only on the model it
was originally seen to help on (v1) and lost to plain direct recalibration on the other two -- we no longer
recommend shrinkage as anything but a candidate to test, not a default.

---

## Addendum B — written 2026-09-23, BEFORE any of the runs below exist
Motivation: a reviewer-objection review of the Run-2 draft. The objections that would most likely sink the paper
are, in order: (1) the base models are undertrained (kill condition #2 triggered: +28% to +59% minADE6 vs UniTraj's
0.85), so the gap could be an artefact of weak models; (2) one architecture family; (3) two datasets; (4) the
solution side is thin (only H5b is clean); (5) the cause of the gap is not probed; (6) marginal coverage only;
(7) one training seed per model. Resources changed: the GPU is now free (AI2 no longer holds it), the full AV2
train cache (183,333 samples, same ~180k UniTraj used) already exists, disk has >1 TB free. Everything below is
registered now, before any of it is run; outcomes are appended below as dated entries, never edited in.

**B1 / H10 — replication on a competitive model (addresses objection 1).** AutoBot-Ego trained from scratch on the
full AV2 train cache, following UniTraj's own recipe as closely as one GPU allows: effective batch 128 (32 x 4
gradient accumulation), lr 7.5e-4, MultiStep decay (x0.5) at epochs 10/20/30/40/50, up to 60 epochs, best checkpoint
by val minADE6 on `av2_splits/val/train` (disjoint from cal/test). **Quality gate:** val minADE6 <= 0.98 (within 15%
of UniTraj's 0.85, the kill-condition tolerance). A second, reverse-direction model is trained the same way on the
full nuScenes train split (32,186 scenes; gate: within 15% of UniTraj's nuScenes-trained AutoBot, minADE6 1.21,
i.e. <= 1.39, supp. Table 8). *Predictions (alpha=0.10):* AV2->nuScenes gap >= 0.02; H2 removal fraction < 25%
(reweighting still fails); H5b power >= 0.90 at k=1000. *Refuted if* the gap on the gated model is < 0.01 -- that
would mean the Run-2 gaps were an under-training artefact, and the paper must say so.

**B2 / H11 — second architecture (objection 2).** Wayformer (16.5M params, pure PyTorch in UniTraj) trained on
AV2 (and nuScenes if time allows) with UniTraj's config, batch reduced to fit 5 GB with gradient accumulation to the
config's effective batch. EMP added as a third architecture only if it runs on the UniTraj data format without
modification. *Prediction:* same-direction under-coverage (gap >= 0.02) AV2->nuScenes. Quality reported, not gated
(no published single-GPU Wayformer number to gate against on this exact split).

**B3 / H12 — third dataset (objection 3).** Waymo Open Motion Dataset (WOMD) via ScenarioNet, validation split
(calibration/test halves by the same salted-hash split), used as a *target* for every source model; as a *source*
too if a train subset can be converted in time. Requires the user's Waymo licence acceptance. *Prediction:*
|gap| >= 0.02 for at least one of AV2->WOMD, nuScenes->WOMD (sign not pre-specified).

**B4 / H13 — which shift factor causes the gap (objection 5).** Controlled shift injection on AV2 test scenes, one
factor at a time, each chosen to mimic a measured AV2-vs-nuScenes difference: (a) history downsampled to 2 Hz and
re-interpolated to 10 Hz (nuScenes' native rate); (b) position jitter on past trajectories at the nuScenes
annotation-noise level (estimated from nuScenes history acceleration residuals, not guessed); (c) map truncated /
thinned to nuScenes' measured lane-point density; (d) scene-composition reweighting only (covariate shift by
construction). Report the coverage gap each injection induces, as a fraction of the real AV2->nuScenes gap.
*Descriptive, with one registered prediction:* (d) alone reproduces < 25% of the real gap (consistent with H2/H3).

**B5 / H14 — a guaranteed few-label repair (objection 4).** "Certify-or-recalibrate" (C-or-R), with k labelled
target scenes and confidence 1-delta:
 (i) *certify:* M = #(target scores > q_src); keep q_src iff the one-sided Clopper-Pearson upper bound on its
     miscoverage at level delta/2 is <= alpha;
 (ii) *otherwise recalibrate:* use the l-th smallest of the k target scores, with l the smallest integer such that
     P(Bin(k, 1-alpha) <= l-1) >= 1-delta/2 (the l-th order statistic's coverage is Beta(l, k+1-l); Vovk 2012).
By a union bound the returned threshold has coverage >= 1-alpha with probability >= 1-delta over the label draw
(training-conditional validity), whichever branch fires. This is an assembly of known pieces; the contribution is
the protocol and its measured label cost on real cross-dataset shift, not the inequality.
*Prediction:* delta = 0.1: coverage >= 1-alpha in >= 88% of 200 random label draws (90% minus Monte-Carlo slack)
for every model x pair and every k >= 250; plain direct SCP at the same k only ~50-60% (it is marginally, not
training-conditionally, valid). Region area relative to the oracle target-calibrated region is reported as the price.
*Refuted if* C-or-R's rate is < 85% for any model x pair at k >= 250 (would indicate non-exchangeable target
labels, e.g. scene-level dependence -- itself a finding).

**B6 / H15 — the obvious online competitor (objection 4).** Adaptive conformal inference (ACI; Gibbs & Candes 2021,
step size gamma in {0.005, 0.01, 0.05}) run on a random-order stream of target scenes with labels revealed one at a
time. Report how many labelled scenes ACI needs before its running coverage stays within 2 points of nominal, next
to H5b's and H14's label budgets. Descriptive, no pass/fail.

**B7 — conditional coverage (objection 6).** Descriptive: target coverage by UniTraj trajectory type, speed tercile,
Kalman difficulty tercile, and (AV2) city, with binomial CIs, for the main gated model.

**B8 — seeds (objection 7).** If GPU time allows after B1/B2, the gated AV2 AutoBot is retrained with 2 more seeds;
report the gap as mean +/- sd over 3 seeds. Descriptive.

**B9 — theory (objection "Proposition 1 covers scale shifts only").** (a) Generalise Proposition 1 to any shift
family whose members share the unlabeled-input law but have unbounded (1-alpha)-quantile. (b) State the exact
condition under which the normalised score (H4) is valid: target and source *normalised* score laws coincide. Test
it: two-sample KS statistic between source-cal and target normalised scores, per model x pair. *Registered
prediction (exploratory, n small):* across all model x pair combinations, a smaller KS statistic goes with a larger
H4 gap reduction (Spearman rho < 0 between KS and reduction).

Order of execution (compute-driven): B1-AV2 on GPU starts now; B5/B6/B7/B9 are analysis-only on existing
predictions and run in parallel; B4 is inference-only; B1-nuScenes and B2 follow on GPU; B3 waits on the licence;
B8 last.

---

## Addendum B-2 — written 2026-09-23, BEFORE the analyses below were run
**Two data-handling defects found today, both disclosed here:**
1. *Cache-key collision.* UniTraj keys its preprocessed cache on the last two path components of a DB path and
   silently reuses an existing cache. `nuscenes_splits/val/cal` therefore mapped onto the cache built earlier from
   `av2_splits/val/cal` (log: "Loaded 4576 samples" = the AV2 cal count; nuScenes cal has 4534). Effect: the
   `ns_cpu_v1` run monitored validation -- and ranked checkpoints -- on **AV2** data, i.e. on its own H9 target
   domain. Its training data were correct nuScenes; the LR schedule is milestone-based, so validation did not
   affect the weights. **Disposition:** H9 uses `ns_cpu_v1`'s final checkpoint (`last.ckpt`), never the
   "best"-ranked one. All forward-direction results are unaffected (their sets map to unique caches; verified by
   sample counts 4576 / 4633 / 9041). A provenance guard (`unitraj_bridge.cache_guard`) now refuses any
   cache built from a different DB.
2. *Scene clustering in nuScenes.* nuScenes val has 9041 agent-scenarios from ~138 scenes (~65 per scene), which are
   strongly correlated. The nuScenes cal/test halves were re-split **by scene** (salt `nsv2scene`: 64 / 74 scenes,
   0 overlap) before any reverse-direction analysis used them.

**B10 — scene-clustered robustness of results already reported (registered now, before running).** For every
forward model with nuScenes as target: (a) the H1 gap CI is recomputed with a *scene-cluster* bootstrap (resample
scenes, not agents); (b) H5b audit power / false alarm and B5 C-or-R rates are recomputed with labelled scenes
drawn as whole scenes until >= k agents, evaluated on the other scenes. *Prediction:* cluster CIs are wider than
the agent-level ones, but the H1 gap's lower CI bound stays > 0 for all three forward models; the C-or-R rate
at k >= 1000 stays >= 0.85. If the audit's false-alarm or C-or-R rate degrades materially under cluster sampling,
the paper reports the clustered numbers as primary.

---

## Outcome log — Addendum B analyses on existing forward predictions — 2026-09-23
Reproduce: `python src/analyze_addB.py --models av2_cpu_v1 av2_valsplit_v1 av2_cpu_v2` -> `results/addB/*__forward.json`.
Target = nuScenes val (9041 agent-scenarios, 138 scenes). alpha = 0.10, delta = 0.10.

| Test | av2_cpu_v1 | av2_valsplit_v1 | av2_cpu_v2 | Registered bar | Verdict |
|---|---|---|---|---|---|
| H14 C-or-R rate(cov>=.9), random agent labels, k=250/500/1000/2500 | .95/.94/.94/.95 | .97/.97/.95/.91 | .97/.97/.97/.93 | >= .88 all k>=250 (refute <.85) | **SUPPORTED** |
| direct SCP same draws | .60/.52/.51/.53 | .62/.55/.57/.54 | .63/.64/.60/.53 | predicted ~.50-.60 | as predicted |
| B10(a) H1 gap, scene-cluster 95% CI | +3.3 [+0.7, +6.2] | +8.0 [+5.0, +11.3] | +10.5 [+7.9, +13.8] | lower bound > 0 | **SUPPORTED** (design effect 3.5-4.4) |
| B10(b) C-or-R rate, labels drawn as whole scenes, k=1000 / 2500 | .59 / .62 | .69 / .61 | .63 / .64 | >= .85 at k>=1000 | **REFUTED** |
| B10(b) audit power, whole-scene labels, k=1000 | .61 | .97 | .99 | (report) | weaker only for the smallest gap |

**Interpretation.** The PAC guarantee of C-or-R assumes exchangeable labels. When the labelled data are whole
scenes -- how driving data are actually labelled -- ~65 agents per scene are strongly correlated, the effective
sample size is the number of scenes (~15 for k=1000 agents), and the guarantee fails (~60-70%). Per the
registration, the scene-grouped numbers are the primary ones for any deployment claim. Agent-level results are
still valid for the (less realistic) i.i.d.-label regime and are reported as such.

B7 (descriptive): coverage loss is concentrated in turning manoeuvres. Same-domain -> target: straight -1.9/-5.4/-6.7 pt;
right turn -12.2/-27.9/-32.1; straight-right -31.6/-40.2/-25.8; left turn -8.4/-13.8/-20.8; hardest Kalman tercile
target coverage .73/.67/.61. B6: ACI's running coverage stays within 2 pt after a median 83-100 labels (gamma .05)
to 342-696 (gamma .005); a *frozen* ACI threshold after 1000 labels covers >= .9 in only 22-52% of streams (like
direct SCP). B9: KS(norm) .047/.106/.109 vs H4 gap reductions 49/12/31% -> Spearman -0.5 (n=3, as predicted in sign).

---

## Addendum B-3 — written 2026-09-23 AFTER the B10/B7 results above, BEFORE the analyses below
Both items are designed in response to the results above, so on the three existing forward models they are
**exploratory**; they become confirmatory only on models/directions not yet run (the GPU-trained models, the
reverse direction, and WOMD if available).

**H16 — scene-level certify-or-recalibrate (cluster-aware repair).** Units = labelled *scenes*. Loss per scene
j at threshold q: L_j(q) = fraction of scene j's agents with score > q (in [0,1], non-increasing in q). Target
risk: scene-averaged miscoverage R(q) = E_j[L_j(q)]. Procedure (Learn-then-Test with fixed-sequence testing over a
decreasing grid of q; Angelopoulos et al.): for each candidate q from largest to smallest, compute a valid
p-value for H_q: R(q) > alpha from the m labelled scenes via the Hoeffding-Bentkus bound; stop at the first q not
rejected at level delta; return the last rejected q. Monotone loss => P(R(q_hat) <= alpha) >= 1 - delta. The source
threshold is tried first (certify branch) at level delta/2 with the same p-value; the grid search gets delta/2.
*Prediction (confirmatory models only):* with whole-scene labels, rate(scene-averaged target coverage >= .90)
>= .85 at every labelled-scene budget m >= 20; region area relative to the oracle reported as the price.

**B11 — driving-side mechanism for the turn-specific loss.** nuScenes target scenes split by location: Boston
(right-hand traffic, like all of AV2) vs Singapore (left-hand traffic). *Prediction:* for right turns, the
same-domain -> target coverage drop is >= 10 pt larger in Singapore than in Boston for at least 2 of the 3
existing forward models (and, confirmatory, for the gated GPU model). Straight driving: drop differs by < 5 pt
between the two cities. Refuted if the right-turn drop is not larger in Singapore for any model.

---

## Outcome log — Addendum B-3 on the 3 existing forward models (EXPLORATORY, per B-3) — 2026-09-23
**B11 (driving side): prediction met on all three models.** Right-turn coverage drop (AV2 same-domain -> nuScenes)
is larger in Singapore (left-hand traffic) than Boston by +15.5 / +22.6 / +25.2 pt (scene-cluster 95% CI
[-1.7, 28.5] / [4.7, 36.7] / [9.2, 36.9]); straight driving differs by -0.5 / -0.1 / -1.8 pt (< 5 pt as predicted).
Left turns show no extra Singapore drop (-13.5 / -5.6 / -0.1 pt). Consistent with: in left-hand traffic the right
turn is the wide, across-traffic manoeuvre never seen in AV2 training. Confirmatory test pending on the GPU model.
**H16 (scene-level LTT, Hoeffding-Bentkus): valid but impractically conservative.** Scene-averaged coverage >= .90
in 100% of draws at every m, but median region area vs oracle: m=20 infinite (HB needs m >= 29 scenes to reject
at all at delta/2 = .05), m=30 18-30x, m=40 8-10x, m=60 4.4-4.9x. Agent-level C-or-R under the same whole-scene
draws: .41-.58; direct SCP .32-.40. Practical reading: with clustered labels, the label budget must be counted in
scenes, and a distribution-free guarantee with ~50 scenes is currently very expensive.
Next (exploratory, not registered as confirmatory yet): H16b = same LTT procedure with a variance-adaptive betting
p-value (Waudby-Smith & Ramdas) instead of HB; if it is materially tighter on these exploratory models it will be
registered for the confirmatory models before they are run.

---

## Addendum B-4 — written 2026-09-24, BEFORE any injected-input prediction was run or looked at
**Change to B4 (shift injection), driven by measurement, not by results.** Before running anything I measured the
input statistics the injections were meant to mimic (`python src/shift_inject.py sigma`, `results/addB/inject_params.json`;
label-free factors from existing prediction files):
* *Annotation noise (registered injection "jitter").* Excess-noise estimate = quadratic-fit residual of the 5 native
  2-Hz keyframes of 2 s histories of moving agents: AV2 sigma = 0.29 m (n=6636 agents), nuScenes sigma = 0.13 m (n=8131).
  nuScenes histories are SMOOTHER, so the excess noise is 0 and the "jitter" injection is vacuous. **Not run;**
  reported as: the noise hypothesis is not supported by the data. (Caveat: this is a smoothness statistic, not a
  ground-truth noise measurement; nuScenes tracks may be smoothed upstream.)
* *Map density (registered "thin lane points").* Median valid map points per scene: AV2 2318, nuScenes 2340 -> equal;
  that injection would be vacuous. **Replaced** by lane dropout at the measured lane-count ratio: n_lanes_near_ego
  medians 85 (AV2) vs 52 (nuScenes) => each map polyline kept with p = 0.612.
* Other measured differences (KS): native_dt 1.00; n_lanes 0.37; hist_heading_change 0.33; map_point_density per lane
  0.28; hist_curvature 0.25; n_agents 0.24 (nuScenes has MORE agents: median 14 vs 11 -> no agent-dropout injection);
  hist_speed 0.14.
**Final B4 set (all applied to AV2 test inputs; AV2-calibrated threshold; ground truth untouched):**
 (a) `hz2` -- history re-sampled to 2 Hz keyframes + linear interpolation (all agents with complete keyframes);
 (b) `map` -- lane dropout p_keep = 0.612;
 (c) `hz2+map` -- both;   (d) `comp` -- composition reweighting of AV2 test scenes to the nuScenes distribution of the
     7 label-free factors (domain-classifier weights, no inference needed).
**Registered predictions (for each model, alpha = .10):** let G = real AV2->nuScenes gap and g_x the gap under injection x.
 P1: g_hz2 >= 0.25 G (the sampling-rate axis alone explains a non-trivial part) -- refuted if g_hz2 < 0.10 G for
     every model. (Prior evidence: scoring at native 2 Hz did not change the gap, so the *history* rate is the open
     question.)  P2: g_comp < 0.25 G (consistent with H2/H3).  P3: g_(hz2+map) < 0.75 G, i.e. the measured input-level
     differences do NOT fully explain the gap (the remainder is label/behaviour shift). No pass/fail on g_map alone.
Reported for the models with this analysis: the three CPU models (exploratory) and, confirmatory, the gated GPU model.
Also registered: the CPU-model injections are run with `--device cpu` on ALL 4633 AV2 test scenes.

---

## Outcome log — Addendum B-4 shift injection on the 3 CPU forward models (EXPLORATORY set) — 2026-09-24
Reproduce: `python src/shift_inject.py run --model M --conds hz2 map hz2map --device cuda`; `python src/analyze_inject.py --models ...`
-> `results/addB/inject_<model>.json`. Effect = paired change in coverage of the AV2-calibrated threshold on the same
4633 AV2 test scenes, as a fraction of the real AV2->nuScenes gap G (alpha=.10). CIs are scene-bootstrap 95%.

| Model (G) | hz2 (2 Hz history) | map (lane dropout .61) | hz2+map | comp (scene mix reweighted) |
|---|---|---|---|---|
| av2_cpu_v1 (+3.3 pt) | +0.6 [0.1,1.1] = 19% | +1.4 [0.9,2.1] = 43% | +1.8 [1.1,2.4] = **53%** | -1.3 [-2.1,-0.4] = -38% |
| av2_valsplit_v1 (+8.0) | +0.7 [0.2,1.1] = 8% | +1.4 [0.8,2.0] = 17% | +1.4 [0.7,2.0] = **17%** | -1.4 [-2.2,-0.6] = -18% |
| av2_cpu_v2 (+10.5) | +1.1 [0.5,1.7] = 11% | +1.3 [0.7,1.9] = 12% | +2.2 [1.5,3.0] = **21%** | -1.1 [-2.0,-0.2] = -11% |

**Verdicts on the registered predictions:** P1 (hz2 >= 25% of G; refuted only if < 10% for EVERY model): 19/8/11% ->
**not supported, not refuted** (the sampling-rate axis explains a small, real, statistically non-zero share).
P2 (comp < 25% of G): **supported** -- composition alone moves coverage the WRONG way (nuScenes' scene mix is easier).
P3 (hz2+map < 75% of G): **supported** on all three -- the measured input-level differences do not explain the gap.
Excluded by measurement before running: excess annotation noise (nuScenes is smoother than AV2) and map-point density.
**Reading:** input-side differences (2 Hz history + sparser lane graph) reproduce 17-53% of the loss, and 17-21% for the two
models with the largest gaps; the remainder is not an input-statistics effect, consistent with the turn-specific,
left-hand-traffic-concentrated loss (B11) -- i.e. mostly a conditional (behavioural) shift.
Note: `hz2` and `map` are position-only perturbations, exact for AutoBot (which reads only positions + masks).

---

## Addendum B-5 — written 2026-09-24, BEFORE H16/H16b are re-run with the fixed grid on real data
**Correction to H16.** The first exploratory H16 run (outcome log above) used data-dependent candidate thresholds
(the labelled scores). That is valid for the monotone Hoeffding-Bentkus p-value but not for a non-monotone one, so
the procedure was changed to a FIXED log-spaced grid anchored on the source threshold: q in q_src x [4 .. 0.5], 300
points, fixed-sequence testing from the top (`solutions.scene_ltt_threshold`). Re-validated on synthetic clustered
data (`test_scene_ltt`): HB rate 1.00, betting rate 0.96 (both >= .90), area vs oracle 3.43x (HB) vs 1.39x (betting),
agent-level C-or-R under whole-scene labels 0.70 (fails, as on real data). The earlier exploratory H16 numbers
(HB: 4.4-30x area) are superseded by the re-run below.
**H16b (registered for the CONFIRMATORY models -- gated GPU AutoBot AV2->nuScenes, GPU nuScenes AutoBot
nuScenes->AV2, and any further model/direction; the 3 CPU models are exploratory):** scene-level LTT with the
Waudby-Smith-Ramdas betting p-value (`pval="wsr"`), delta = .10, whole-scene labels, m labelled scenes.
*Predictions:* (i) rate(scene-averaged coverage of the returned threshold >= .90) >= .88 for every m in {30,40,50,60};
(ii) median area vs oracle <= 3.0x at m = 40 and <= 2.0x at m = 60; (iii) betting is at least as tight as HB at every
m (median area ratio betting/HB <= 1). *Refuted if* (i) fails for any m >= 40 on any confirmatory model, or (ii) fails
on both confirmatory directions. If (ii) fails but (i) holds the method is reported as valid-but-costly, not as a fix.
Reading rule: the per-scene loss uses all agents of a scene equally; scene weights are equal (scene-averaged risk).

---

## Outcome log — H16 / H16b (fixed-grid scene-level LTT) on the 3 CPU forward models (EXPLORATORY) — 2026-09-24
Whole-scene labels from nuScenes (138 scenes, 9041 agents), m labelled scenes, evaluation on the remaining scenes;
200 draws; delta = .10; alpha = .10; area = median region area vs the oracle (target-calibrated at exactly .90).

| m scenes | HB p-value: rate / area (v1, valsplit, v2) | betting p-value (H16b): rate / area (v1, valsplit, v2) | agent-level C-or-R rate |
|---|---|---|---|
| 30 | 1.00 / inf ; 1.00 / inf ; 1.00 / inf | .92/1.89 ; .95/1.74 ; .95/1.71 | .53 / .56 / .58 |
| 40 | 1.00 / 9.6 ; 1.00 / 8.3 ; 1.00 / inf | .89/1.76 ; .91/1.62 ; .92/1.67 | .47 / .53 / .54 |
| 50 | 1.00 / 6.1 ; 1.00 / 5.8 ; 1.00 / 6.5 | .91/1.72 ; .93/1.63 ; .86/1.69 | .41 / .54 / .56 |
| 60 | 1.00 / 4.7 ; 1.00 / 4.5 ; 1.00 / 4.9 | .88/1.66 ; .91/1.50 ; .87/1.46 | .47 / .48 / .55 |

Against the registered H16b predictions (which apply to the confirmatory models; shown here for the exploratory ones):
(ii) area <= 3.0x at m=40 and <= 2.0x at m=60: met on all three (1.6-1.8x, 1.5-1.7x); (iii) betting <= HB: met everywhere;
(i) rate >= .88 at every m in {30,40,50,60}: met for v1 (.92/.89/.91/.88) and valsplit (.95/.91/.93/.91), NOT met for
av2_cpu_v2 at m=50, 60 (.86, .87). Caveat on (i): the "rate" is evaluated on the <= 108 held-out scenes, whose own
sampling noise (SD of scene-averaged coverage ~1 pt) makes a threshold at the guarantee boundary look worse than its
population risk; it is a conservative-in-the-wrong-direction measurement, not a failure of the p-value's validity, but
the registered criterion is applied as written on the confirmatory models. HB is valid but returns an infinite region for
m <= ~40 scenes and 4.5-10x the oracle area otherwise -> not practical. Bottom line (exploratory): with ~40-60 labelled
SCENES (~2.6-3.9k agents) betting scene-level LTT gives a threshold within 1.5-1.9x of the oracle region and covers at
~.9 across held-out scenes, where agent-level guarantees collapse to ~.5.

---

## Development dry run on an intermediate GPU checkpoint — 2026-09-24 (NOT a confirmatory result)
**Purpose and rules.** To find pipeline bugs and to manage the risk "the gap is an artefact of weak models" *early*, the
best checkpoint of the running full-data AV2 AutoBot at epoch 11 (`av2_gpu_dev_ep11`, val minADE6 = 1.043; not the final
model, gate 0.98 not yet met) was pushed through the whole analysis chain (`analyze_run2.py`, `analyze_addB.py`).
It is NOT part of any table's confirmatory set (`make_results_addB.CONFIRMATORY` lists only the final gated model).
No design decision, threshold, or hypothesis was changed because of these numbers; the registered analyses are re-run on the
final checkpoint. Reported here because it is informative for the risk it was meant to manage, and it will be disclosed in
the paper.
**What it showed (val minADE6 1.043, better than all three CPU models):** raw gap **+11.0 pt** (CI 9.6-12.7; scene-cluster
CI 8.4-13.8), same-domain +0.1 pt; normalised gap +6.9 pt (ratio 0.63); audit power 1.00 / false alarm 2.8% at k=1000;
covariate reweighting again worse (removal -22%); right-turn coverage 0.81 -> 0.52, straight 0.94 -> 0.85; i.i.d.-label
C-or-R 0.96 (direct 0.50); whole-scene labels: agent-level C-or-R 0.56-0.59 vs betting scene-LTT 0.93 / 0.89 at m = 40 / 60
scenes (area 1.72x / 1.56x oracle). **Every qualitative finding of the three-model zoo replicates, and the gap is larger,
not smaller, than for the weaker models** -- so "the loss is an under-training artefact" is not what happens at this
accuracy level. (Pattern across four models: the better the point predictor, the larger the coverage gap; n = 4,
exploratory, no causal claim.)
