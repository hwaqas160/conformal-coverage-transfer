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

---

## Addendum C — third dataset (Waymo Open Motion) — written 2026-09-28, BEFORE any Waymo prediction exists or was looked at
**Why.** Two datasets invite "is this a property of AV2/nuScenes or of dataset shift?". A third dataset makes the claim about
shift in general. Data: Waymo Open Motion v1.2.1 `scenario/validation`, first 30 of 150 shards (8,610 scenarios; 287 in shard 0),
converted with ScenarioNet through UniTraj's own Waymo pipeline, **unmodified** (whatever history length/padding UniTraj
produces for Waymo is part of the shift; nothing is tuned to Waymo).
**Design (fixed now).** Scenario-disjoint calibration/test split of the Waymo scenarios by salted hash (salt `wm1`, 50/50 by
scenario id; all agents of a scenario go to the same side). Forward direction only, using the confirmatory model
`av2_gpu_full` (last checkpoint, per B-2) calibrated on the AV2 calibration half. If the reverse-direction model
`ns_gpu_full` passes its gate, it is also applied to Waymo (nuScenes -> Waymo) as a second source. No Waymo-trained model is
trained (none is needed: the questions concern a source-calibrated forecaster meeting a new dataset). All uncertainty
statements use scenario-level resampling. The exploratory CPU zoo is NOT run on Waymo (no gate, avoids a forking path).
**Hypotheses (predictions written before data).**
- H17 (breaks on a third dataset): source-only coverage on Waymo is below 0.90 by at least 3 pt (raw gap >= +3.0 pt, scenario-cluster
  95% CI excluding 0). Refuted if the gap is < 3 pt or the CI includes 0. (Direction is not assumed obvious: Waymo history is
  shorter, which alone could go either way for coverage; a *negative* gap, i.e. over-coverage, would be reported as such and the
  paper's claim would be restated as "not guaranteed", not "under-covers".)
- H18 (generality): raw gap > 0 on both target datasets (nuScenes and Waymo) for the same source model, each with CI excluding 0.
  If Waymo shows over-coverage the paper says the *sign* is dataset-dependent and only the loss of the guarantee is general.
- H19 (repair transfers): scene-level betting LTT, with m in {30,40,50,60} labelled Waymo scenarios drawn 200 times, achieves
  rate(scene coverage >= 0.90) >= 0.88 at every m, and median area <= 3.0x oracle at m = 40. The number of predicted agents per
  Waymo scenario is smaller than in nuScenes (design effect smaller), so the agent-level C-or-R baseline may *not* fail badly here;
  no failure threshold is asserted for it. Whatever it does is reported (if it is fine on Waymo, the paper says that scene-level
  labelling matters when scenes carry many correlated agents).
- H20 (mechanism consistency, exploratory): as on nuScenes, the normalised score halves or reduces the gap without ever
  removing it entirely; reported without a threshold.
**Reporting commitment.** All four are reported whichever way they land; Waymo results are labelled confirmatory (registered
before data) but only for the analyses listed here, every other Waymo analysis is exploratory.

---

## Outcome log — Addendum C (Waymo, confirmatory: av2_gpu_full and ns_gpu_full, the same final models used for the
main H1/H4 confirmatory result) — 2026-09-29
**H17 (breaks on a third dataset): SUPPORTED, both directions.** AV2->Waymo: coverage 66.6% (target 90%), raw gap
+23.4 pt, scenario-cluster 95% CI [21.6, 24.9] (same-domain AV2 control: +0.4 pt). nuScenes->Waymo: coverage 71.7%,
gap +18.3 pt, CI [16.9, 20.4] (same-domain nuScenes control: -1.9 pt, i.e. slight over-coverage). Both gaps are LARGER
than the corresponding nuScenes/AV2 cross-dataset gaps for the same source models (+15.1 pt and +7.7 pt respectively) --
Waymo is the hardest target of the three, consistent with it having the largest native-frequency and history-length
difference from both training datasets.
**H18 (generality across two real targets, same source model): SUPPORTED, both directions.** av2_gpu_full: +15.1 pt
(nuScenes) and +23.4 pt (Waymo), both positive with CIs excluding 0. ns_gpu_full: +7.7 pt (AV2) and +18.3 pt (Waymo),
both positive. No sign flip was observed on either target; the paper can state the direction (under-coverage, not just
"loss of guarantee") as a general finding across four source->target pairs, not merely the original AV2<->nuScenes pair.
**H20 (normalisation reduces but does not remove): SUPPORTED, both directions.** av2_gpu_full: raw +23.4 pt -> normalised
+16.7 pt (29% reduction). ns_gpu_full: raw +18.3 pt -> normalised +16.3 pt (11% reduction, smaller than on nuScenes).
CI lower bound stays > 0 in both cases -> not removed.
**H19 (scene-level fix transfers to Waymo): PARTIALLY SUPPORTED, weaker than on nuScenes.**
  av2_gpu_full -> Waymo, betting scene-LTT rate(scene cov >= .90): m=30 .90, m=40 .88 (meets the pre-registered >=.88
  bar), m=50 .85, m=60 .86 (BELOW .88). Area vs oracle: 1.86x/1.58x/1.56x/1.67x (meets <=3.0x at m=40 easily).
  ns_gpu_full -> Waymo: m=30 .85, m=40 .82, m=50 .81, m=60 .81 -- BELOW .88 at every m; area 2.02x/1.78x/1.62x/1.73x
  (still meets the area bar). Registered criterion (i) (rate >= .88 at every m in {30,40,50,60}) is therefore NOT met in
  full on Waymo, unlike on the nuScenes target where the exploratory zoo mostly met it. Reading: the betting LTT
  procedure remains VALID (it is a proof, not an estimate -- population coverage is still guaranteed at 1-delta by
  construction) but empirically conservative-in-the-wrong-direction here is not what is happening; rather Waymo's
  design effect (fewer, more heterogeneous agents per scenario: mean 3.7 vs nuScenes' ~65) combined with only 200
  draws over ~4300 scenarios gives more between-draw variance in the *empirical* rate than on nuScenes, and 60-88%
  measured rate is still far above the un-repaired baseline (H17 gap of 18-23 pt corresponds to ~0% of draws meeting
  90%). **Reported honestly as a partial replication**: the fix clearly helps on Waymo (raises coverage from ~70% to
  ~90% on average) but the specific numeric bar written into H19 before this run is not met at m in {50,60} for either
  source model. The paper states the m=30/40 result as the confirmed operating point and reports m=50/60 as a
  measured shortfall, not a hidden one.
**Bottom line for Addendum C.** The central claim (the guarantee breaks under dataset shift and gets worse, not better,
away from the training distribution) generalises to a third, independently collected dataset with a different sensor
suite and city set. The fix generalises partially: it works at the same label budget (30-40 scenes) that was validated
on nuScenes, but the paper will not claim it holds at 50-60 scenes on Waymo without qualification.

---

## Outcome log — H16b on the confirmatory model (av2_gpu_full, gate-passing, AV2->nuScenes) — 2026-09-29
Registered criterion (Addendum B-5): (i) rate(scene cov >= .90) >= .88 at every m in {30,40,50,60}; (ii) area <= 3.0x
oracle at m=40 and <= 2.0x at m=60; (iii) betting <= Hoeffding-Bentkus area.
**Result:** betting rate 93%/92%/84%/87% at m=30/40/50/60; area 1.86x/1.86x/1.81x/1.81x. Criterion (i) is met at
m=30,40 (93, 92 >= 88) and NOT met at m=50,60 (84, 87 < 88). Criterion (ii) is met (1.86x <= 3.0x at m=40; 1.81x <=
2.0x at m=60). Criterion (iii) is met throughout: HB is valid but returns an infinite region at every m tested here
(96.5-100% of draws), so betting is the only practical choice. Agent-level C-or-R and direct SCP remain far below any
usable rate at m=40 (50% and 36.5%), so the qualitative finding -- whole-scene labelling needs a scene-level, not
agent-level, guarantee -- is confirmed without qualification.
**Reading: same shape of partial result as Addendum C's H19 on Waymo** (met at the smaller label budgets, missed at
the larger ones, on the SAME confirmatory model and the SAME registered numeric bar). This is now three independent
runs of the identical criterion (nuScenes exploratory zoo mostly passing all m, nuScenes confirmatory model passing
m=30/40 only, Waymo confirmatory passing m=30/40 only for one of two source models): the fix is real and large
(un-repaired baseline is 36.5% vs 84-93% repaired) and the practical operating point the paper recommends is
**30-40 labelled scenes**, not 50-60. Recommending m up to 60 without qualification, as the exploratory-zoo text
originally implied, is not what the confirmatory evidence supports, and the paper is corrected accordingly.

---

## Addendum D -- revision R1: defect declaration, corrected evaluation design, new analyses -- written 2026-09-30, BEFORE any corrected rerun
**Status of earlier entries.** The outcome logs for H16, H16b (Addendum B-5 and its confirmatory outcome) and H19 (Addendum C)
stay exactly as written and are SUPERSEDED for the betting-LTT rows by the results logged under this addendum. HB-LTT,
agent-level C-or-R, direct SCP, and all coverage-gap results (H1, H4, H17, H18, H20) use different code paths and are unaffected.

**D1. Defect.** `solutions.wsr_pvalue` capped the bet at lam <= 0.5/alpha (= 5). The test is H0: E[L] >= alpha with payoff
(alpha - L), L in [0,1]. Capital K = prod(1 + lam_i (alpha - L_i)) stays non-negative only if lam_i <= 1/(1-alpha) (= 1.11), so
with the old cap the capital could go negative and Ville's inequality no longer applied. Reproduced by `src/diag_betting_capital.py`
on av2_gpu_full: rejections made while capital was negative -- Waymo m=40: 6346, m=60: 8842; nuScenes m=40: 2685, m=60: 5647.
Found in an internal review before submission. Not visible earlier because the synthetic test used many agents per scene.
**D2. Fix (fixed now, not tuned).** lam_i <= 0.5/(1-alpha); bets stay predictable (past losses only); assert capital >= 0.
**D3. Evaluation design (fixed now).** Labelled pool and evaluation pool are disjoint by scene:
  - AV2 -> nuScenes: labelled pool = nuScenes TRAIN scenes (predicted with the AV2 model, which never trained on nuScenes);
    evaluation pool = all 138 nuScenes val scenes.
  - AV2 -> Waymo and nuScenes -> Waymo: labelled pool = `wmcal` scenarios, evaluation pool = `wmtest` scenarios.
  - nuScenes -> AV2: AV2 clips are single-agent, so the scene rule equals the agent rule; labelled pool = AV2 cal half, evaluation
    pool = AV2 test half.
  - m in {20,30,40,60,80,100} labelled scenes, 1000 draws per m. For each draw the threshold is computed from the m labelled
    scenes and R(q_hat) is evaluated EXACTLY on the full evaluation pool (scene-averaged miscoverage). Reported: P[R(q_hat) <= alpha]
    with a Wilson 95% interval, and the median area ratio vs the oracle threshold of the evaluation pool.
**D4. Success criterion (fixed now).** Validity: for every m tested, point estimate of P[R <= alpha] >= 0.88 AND Wilson lower
bound >= 0.85. Recommended budget: the smallest m whose median area ratio <= 2.0. If validity fails after the fix, the paper drops
the betting variant and reports the Hoeffding-Bentkus variant only. If the exact synthetic validity test (D5) fails, no repair
claim is written until it passes.
**D5. Regression tests (fixed now).** (a) adversarial scenes with L in {0,1} never give negative capital; (b) exact-superpopulation
validity: scenes drawn i.i.d. from a known distribution, true R(q) analytic, P[R(q_hat) <= alpha] >= 1 - delta - 2 SE over 2000
draws for m in {20,40,60,100}; (c) a mutation check: restoring the old cap must make (a) fail.
**D6. New analyses (each labelled).**
  - H21 history confound (EXPLORATORY-MECHANISM): inject "AV2 history truncated to 1.1 s" (mask frames older than 1.1 s, as UniTraj
    does for Waymo) into AV2 test inputs of av2_gpu_full and the 3 exploratory models; report coverage lost as a share of the
    AV2 -> Waymo gap. Prediction: at least 25% of the gap. No claim is made if it is below 10%; whatever it is gets reported.
  - H22 hierarchical baseline (CONFIRMATORY comparison): scene-level split CP after Dunn et al. 2023 -- (a) one random agent per
    labelled scene, standard split CP quantile; (b) the pooled agent quantile. Prediction: neither reaches the D4 validity criterion
    at m = 40 on nuScenes, and both are reported next to betting-LTT and HB-LTT.
  - H23 mechanism (DESCRIPTIVE, n = number of models): q_src, q_oracle (metres) and q_oracle/q_src per model and target, against
    minADE6; score CDFs source vs target. No test is run with n < 8.
  - T-IV relevance (DESCRIPTIVE): region radius in metres (source vs repaired) and share of agents outside the region, by manoeuvre.
  - Statistics: Wilson 95% intervals on every reported rate. Multiplicity: the primary confirmatory claims are the four
    source->target gap tests (H17/H18), each with a cluster-bootstrap interval excluding 0 by a wide margin; Holm adjustment would
    change no conclusion, and this is stated in the paper. All other analyses are descriptive or labelled exploratory.
**D7. Release.** The published Zenodo record (v1, 10.5281/zenodo.23054413) contains the defective function. A v2 with a changelog
naming D1 will be published, and the paper will cite the concept DOI.

---

## Addendum D-2 -- disclosure of the fix's effect on usefulness, and extension of the label-budget grid -- 2026-09-30, BEFORE the D3 rerun
**What the corrected code does (preview, NOT the registered D3 evaluation).** A 100-draw check of the corrected betting rule on the
OLD design (av2_gpu_full, AV2 -> nuScenes val, labels drawn from the same 138 scenes) returned an infinite threshold in 100% of draws at
m = 30, 40 and 60 (Hoeffding-Bentkus was already infinite there). The earlier finite thresholds (1.6-1.9x oracle at m = 30-60,
Addendum C outcome log and the confirmatory H16b entry) were therefore produced by the defect in D1, not by evidence. Those entries
stay unedited and are superseded.
**Why (mechanism).** With bets limited to lam <= 1/(1-alpha) ~ 1.1 (needed for a non-negative capital when L in [0,1]) and per-scene
losses with large between-scene spread, log-capital grows by only ~0.03 per labelled scene when the tested threshold is slightly
better than alpha; certification at level delta/2 = 0.05 needs log-capital ~3.0, i.e. on the order of 100 or more scenes.
**Consequence for the registered design.** The D3 grid m in {20,30,40,60,80,100} would be all-infinite and uninformative for the
budget question. Extended BEFORE the run to m in {20,30,40,60,80,100,150,200,300} (the nuScenes train pool has ~700 scenes and Waymo
wmcal ~4300, so all m are feasible; at m >= 300 on nuScenes the evaluation pool stays the 138 val scenes). D4 is unchanged: validity
criterion for every m tested, recommended budget = smallest m with median area <= 2.0x oracle, reported as "none within the tested
range" if no m qualifies. Nothing about c = 0.5 (D2) or the criteria is changed after seeing results.
**Additional pre-registered option (fixed now, reported either way).** Because a strict guarantee at alpha may need more scenes than a
deployer can label, the analysis also reports the same rules at a relaxed target alpha' = 0.12 (i.e. guaranteeing scene-averaged
coverage >= 0.88 at probability 0.90) as a sensitivity row. It is labelled a tolerance variant, never presented as the alpha = 0.10
guarantee.

---

## Addendum E -- second architecture (Wayformer), revision R1 step 2 -- written 2026-09-30, BEFORE any Wayformer training
**2.1 Pretrained checkpoints.** Checked the UniTraj upstream repository (README, releases) for an official pretrained AV2 or
nuScenes checkpoint (Wayformer or MTR): none is offered. Training from scratch is the only option; recorded here so the
decision is not re-litigated later.
**Model and recipe.** Wayformer (UniTraj's implementation, `configs/method/wayformer.yaml`): 256 hidden, 2 encoder / 8 decoder
layers, OneCycleLR (max_lr 2e-4, div_factor 100, final_div_factor 10), AdamW. Same full 183k-scene AV2 train cache, same
val_db (`av2_splits/val/train`) as av2_gpu_full, so the two architectures are compared on identical data and splits. Epochs
set by the benchmark (2.3) to complete a FULL one-cycle schedule within roughly 60 GPU-hours; never stopped early, since
OneCycleLR's schedule is defined by the epoch count set at launch.
**Gate (fixed now).** val minADE6 <= 0.98 (within 15% of UniTraj's reported Wayformer-on-AV2 result), identical rule and
tolerance to av2_gpu_full's gate, checked against UniTraj's Table 8 Wayformer row (to be re-verified from the paper's own
reference list; if UniTraj's Wayformer/AV2 number differs from AutoBot's 0.85, THIS gate is recomputed before training
finishes, and the recomputation is logged here before the model is evaluated against it).
**If the gate is missed (2.6, fixed now):** one extension only, +50% epochs as a fresh complete one-cycle schedule
restarted from the last weights (not a partial resume that breaks the schedule). If still missed, Wayformer is reported as
a second architecture with its measured accuracy stated plainly, labelled exploratory for the point-accuracy comparison,
and every downstream analysis on it is still run and reported -- a missed gate is a disclosed limitation, not a reason to
withhold results.
**What is run on it once trained (registered now):** the same confirmatory battery as av2_gpu_full -- H1/H4 (AV2->nuScenes,
AV2->Waymo), H17/H18/H20 (Waymo), and the corrected scene-level repair (Addendum D design, D5-tested code) on both targets.
Injection experiments (Sec. inject) need velocity/acceleration recomputed for Wayformer, since it reads all object-trajectory
features rather than position alone; this is implemented and unit-tested against AutoBot (must reproduce existing AutoBot
injection numbers exactly) BEFORE it is run on Wayformer, so the two architectures cannot be run on inconsistent injection code.
**Benchmark (2.3, to run before committing to an epoch count):** `--profile_steps 50 --method wayformer` at batch 16 and 32;
s/step, peak memory; if OOM, batch 16 with grad accumulation 8 (effective 128, matching the AutoBot recipe).

---

## Addendum E-1 -- Wayformer gate verified against the primary source -- 2026-09-30
Checked directly (arXiv:2403.15098v1, PDF pages 21-24): UniTraj's Table 8 is continual learning (WOMD<->AV2 fine-tuning,
AutoBot only, brier-minFDE) -- NOT a per-architecture accuracy table. The correct source is **Table 15** (Appendix B.2,
"Cross-dataset generalization ... minADE metric"), Argoverse-2-trained / Argoverse-2-evaluated column (diagonal), K=6
modes (confirmed minADE = minADE6 by their Sec. 3.3 definition): **MTR 0.85, Wayformer* 0.85, AutoBot 0.85** -- all
three architectures report exactly 0.85 in UniTraj's own reimplementation. (Wayformer is marked with an asterisk because
UniTraj re-implemented it: "the original code has not been released", Sec. 3.2 -- the same situation we are in, so the
comparison is apples-to-apples.)
**Consequence:** Wayformer's gate is identical to AutoBot's, val minADE6 <= 0.98 (within 15% of 0.85), with no
recomputation needed. Addendum E's "to be re-verified" clause is resolved: the gate stated there was already correct by
coincidence, now confirmed rather than assumed. refs.bib / refs_status.md unaffected (feng2024unitraj already verified);
this only corrects the internal table citation ("Table 8" -> "Table 15") for the paper's own setup.tex wording, which
does not currently name a table number and needs none added, but any future draft that cites one must say Table 15.

---

## Outcome log -- Addendum D5 regression tests (exact synthetic validity) -- 2026-10-01
All three tests pass (`results/rev1/test_betting_validity.log`, `src/tests/test_betting_validity.py`):
(a) adversarial {0,1}-loss scenes never drive the betting capital negative; (c) restoring the old 0.5/alpha cap
makes (a) fail (mutation check confirms the test is sensitive); (b) exact-superpopulation validity holds at every
m in {20,40,60,100,200} and every q_src tested (badly wrong / borderline / already valid): P[R(q_hat) <= alpha]
is 0.991-1.000 against a required bound of 0.887, over 2000 draws each (30 cells, all OK). The corrected code is
therefore exactly valid in the regime it is used in. Operationally: betting-LTT returns a finite threshold in 0%
of draws at m=20 and m=40 (matches the preview in Addendum D-2), ~98-100% at m=60, 100% at m>=100; once finite,
median area is 1.4-3.4x the oracle region, falling as m grows (3.3-3.4x at m=60 -> 1.4-1.5x at m=200). Step 1.3 is
complete; proceeding to 1.6 (corrected scene-repair rerun on real data, disjoint pools).

---

## Outcome log -- Addendum D3/D4 -- corrected scene-level repair, disjoint pools, real data (confirmatory) -- 2026-10-01
All four source->target pairs (av2_to_nuscenes, av2_to_waymo, ns_to_waymo, ns_to_av2), confirmatory models
(av2_gpu_full, ns_gpu_full), 1000 draws per m, m in {20,30,40,60,80,100,150,200,300}, labelled and evaluation
pools disjoint by scene as fixed in D3. Full tables: `results/rev1/scene_<pair>.json`.

**D4 validity criterion (point >= 0.88 AND Wilson lower bound >= 0.85) is met by the corrected betting-LTT rule
AT EVERY SINGLE m TESTED, on all four pairs, with no exception.** Point estimates range 0.943-1.000; the single
closest case is ns_to_waymo at m=300 (0.943, Wilson [0.927,0.956], still comfortably above the 0.85 floor). This
fully resolves the external review's #1 finding (measured repair rate below 1-delta): that was entirely the D1
code defect. The corrected, properly-evaluated (disjoint-pool) method is exactly valid, as D5's synthetic check
already showed and this real-data run confirms.

**What changes: the recommended label budget, not the validity.** At small m (20-40) the rule is valid but
useless -- it returns an infinite (uncertifiable) threshold in 100% of draws, because log-capital grows too
slowly under the corrected, conservative bet cap to certify anything from so few scenes. Thresholds start
reliably finite around m=100-150 and the median area (vs. the evaluation-pool oracle) falls as m grows:
  - av2_to_nuscenes: inf=0.06 at m=100, inf=0.00 from m=150 (area 1.91x); area 1.62x at m=200.
  - av2_to_waymo:    inf=0.21 at m=100, inf=0.01 at m=150 (area 1.71x); area 1.51x at m=200.
  - ns_to_waymo:     inf=0.07 at m=100, inf=0.00 at m=150 (area 1.60x); area 1.45x at m=200.
  - ns_to_av2:       inf=0.03 at m=100, inf=0.00 at m=150 (area 2.14x, >2.0x); needs m=200 for area <=2.0x (1.85x).
Applying D4's recommended-budget rule (smallest m with median area <= 2.0x oracle AND infinite-fraction <= 0.05,
the latter clause added here as the natural reading of "a usable region most of the time", consistent with D4's
intent though not spelled out as a separate number): **the practical operating point is m = 150 labelled scenes
for three of the four pairs, and m = 200 for nuScenes -> AV2** (AV2's agent-sparse "scenes", one agent each,
carry less information per labelled unit than nuScenes/Waymo's multi-agent scenes). This supersedes the
exploratory-zoo and pre-D1-fix estimate of "30-40 scenes" everywhere it was stated; that number was an artifact
of the betting capital defect (D1), not a property of the method. The paper is corrected accordingly.

**Hoeffding-Bentkus comparison (also corrected):** at m=150/200 HB is valid throughout too (as it always was --
D1 never affected HB) and its area is now only modestly larger than betting's (e.g. av2_to_nuscenes m=200:
HB 2.11x vs bet 1.62x; ns_to_av2 m=200: HB 1.71x vs bet 1.85x, i.e. HB is slightly BETTER there). Betting's
advantage over HB is real but smaller than the pre-fix numbers suggested.

**New baselines (addressing the external review's missing-baseline point, D6/H22):**
  - Agent-level certify-or-recalibrate (whole-scene-pooled labels, agent-level guarantee): fails badly only for
    av2_to_nuscenes (valid rate falls from 0.58 at m=20 to 0.04 at m=300, the known nuScenes design-effect
    story, ~65 correlated agents/scene). For the other three pairs it stays mostly valid (0.65-1.00), because
    their agents-per-scene ratio is much smaller (Waymo ~3.7/scenario; AV2 reverse-direction scenes are
    single-agent by construction). This sharpens the paper's mechanism claim: agent-level guarantees fail
    specifically when agents-per-scene is large, not as a universal property of "scene-structured" labels.
  - One-agent-per-scene hierarchical split CP (Dunn et al. 2023 style, H22): valid at small m by wide margins
    (noise), but degrades below the 0.88 bar at larger m on every pair (e.g. av2_to_nuscenes 0.57 at m=300,
    ns_to_waymo 0.45 at m=300) -- it is not a reliable alternative to scene-level LTT at the budgets that matter.

---

## Addendum F -- geometric relevance metric: design fixed BEFORE any computation -- written 2026-10-02

A second external review flagged a real risk ahead of time: the "downstream relevance" number proposed in the
revision plan (region radius in metres; share of agents whose true path falls outside a repaired region) is
naturally **agent-weighted** (one count per agent), while Proposition 3 and the scene-level LTT procedure control
the **scene-weighted** risk $R(q)=\mathbb E_j L_j(q)$ (Addendum D / Sec.~method, "scene-level risk control"). These
are different estimands, exactly as flagged for $R(q)$ vs.\ the agent-pooled miss rate. Fixed now, before the metric
is computed, so it cannot be framed after the fact to look more connected to the formal guarantee than it is:

**H24 -- geometric relevance (DESCRIPTIVE only, no guarantee claimed).**
- **Definition.** For a fixed, already-chosen threshold $\hat q$ (the source threshold, or a repaired one from
  Sec.~repair-scene), report (a) the region radius in metres; (b) the agent-weighted share
  $\frac1{\sum_j n_j}\sum_{j,i}\mathbf 1[s_{ji}>\hat q]$ of individual agents whose true path falls outside the
  region. (b) is the agent-pooled miss rate already named and distinguished from $R(q)$ in Sec.~method; this
  addendum only fixes that the SAME distinction applies here, before the number exists.
- **What this is NOT.** Neither quantity is a quantity Proposition 2 or 3 bounds. Proposition 3 bounds
  $\Pr[R(\hat q)\le\alpha]$, the scene-weighted average. The agent-weighted share in (b) can differ from $1-R(\hat q)$'s
  complement whenever scenes have unequal agent counts (nuScenes ~65/scene, Waymo ~3.7/scenario, AV2 1/scene), which
  they do. No sentence in the manuscript may say the scene-level guarantee "covers", "implies" or "guarantees" the
  agent-weighted number; the two are reported side by side, each labelled with its own weighting, and the gap between
  them (if any) is itself reported as a number, not elided.
- **Notation (fixed now).** The scene-weighted guaranteed quantity keeps using $R(q)$ / $\hat q$ as already defined.
  The new descriptive quantity uses a distinct symbol, $M(\hat q)$ (agent-weighted miss share), never $R$, never
  presented in the same inline sentence as a $\Pr[\cdot]\ge 1-\delta$ statement without the word "descriptive" or
  "no guarantee" attached.
- **Status.** Exploratory/descriptive in every case; not run on any model as of this addendum. When it is computed,
  the outcome (including the size of the gap between $M(\hat q)$ and $1-R(\hat q)$'s complement, which is itself an
  informative number about how unevenly agents are distributed across scenes) is logged here before being written
  into the paper.

---

## Outcome log -- Addendum F (descriptive, no guarantee claimed) -- 2026-10-05
Computed by `src/analyze_addF.py` for the confirmatory models at the standard source split-CP threshold (q at 90% of the
source calibration scores). R = scene-weighted miss rate (the quantity Prop. 3 controls); M = agent-weighted miss share.
- AV2->nuScenes: R=0.269, M=0.253 (scenes 138, agents 9041; agents/scene median 53, mean 65, max 288)
- AV2->Waymo:    R=0.323, M=0.335 (scenes 4318, agents 16096; median 3, max 8)
- nuScenes->AV2: R=0.181, M=0.181 (single-agent scenes, so R=M by construction)
- nuScenes->Waymo: R=0.279, M=0.284
**Reading.** The two weightings differ by at most 1.6 points, in opposite directions on different datasets, so the
scene-vs-agent distinction changes the numbers only slightly here; all four give miss rates 2-3x the 10% nominal, and the
qualitative conclusion is identical under either weighting. Note for the text: nuScenes' agents-per-scene is mean 65 but
median 53 with a heavy tail (max 288), so "about 65" should be stated as "mean 65 (median 53)". No claim is made that
Prop. 3 covers M; M is reported side by side with R only. Descriptive result; reported as such.

---

## Outcome log -- Addendum E, Wayformer gate -- 2026-10-07
Training finished (27/27 epochs, `results/ckpts/av2_wayformer/DONE`). check_val_every_n_epoch=2 landed on odd
epochs (1,3,...,25), so the true final checkpoint (`last.ckpt`, end of epoch 26) was never validated during
training; the best LOGGED checkpoint was epoch 25 (val minADE6 0.955). Per setup.tex's registered rule ("the final
model is the last checkpoint, fixed in advance"), `last.ckpt` is the model used downstream, not epoch 25, so its
actual validation score was measured directly (`src/validate_ckpt.py`, same val split/subset/seed as training,
2000 of 13770 AV2 val scenes, seed 0): **val minADE6 = 0.9275**, val minFDE6 = 2.006, miss rate 0.378.
**Gate (Addendum E/E-1): val minADE6 <= 0.98, reference 0.85. PASSED, with 9% headroom** (0.9275 vs 0.98), and
closer to UniTraj's reference than av2_gpu_full's own confirmatory result (0.859/1.084 depending on metric
definition -- see Addendum E-1 for why these two metrics, minADE6 vs the 5-mode/2Hz diagnostic, are not the same
number). No extension (2.6) is needed. Proceeding to post-processing (predictions, run2, addB, addC) on
`av2_wayformer`/`last.ckpt`.

---

## Outcome log -- Addendum E, Wayformer confirmatory results -- 2026-10-07
All pre-registered confirmatory tests (H1/H4, H17/H18/H20, corrected scene-repair design D3/D4) run on
`av2_wayformer` (last.ckpt, gate passed per the entry above). No reverse-direction (nuScenes-source) Wayformer
model was trained, so H9/H18's reverse-target check is not available for this architecture (reported as such,
not estimated).

**H1 (AV2->nuScenes).** Same-domain coverage 89.9% (CI crosses 0 on the gap, as expected). Target (nuScenes)
coverage 83.6%, raw gap **+6.4 points** (CI [5.0, 7.9]), smaller than av2_gpu_full's +15.1 points on the SAME
transfer. Normalised gap +1.3 points (80% reduction) -- much larger repair effect than AutoBot's 27% reduction on
this transfer.
**H17/H20 (AV2->Waymo).** Target coverage 65.0%, raw gap **+25.0 points** (CI [23.5, 26.9]), slightly LARGER than
av2_gpu_full's +23.4 points on the same transfer. Normalised gap +8.4 points (66% reduction).
**Reading: architecture changes the magnitude on nuScenes a lot (15.1 -> 6.4 pt) but barely on Waymo (23.4 -> 25.0
pt), while the direction (under-coverage, never over) and the general shape (Waymo harder than nuScenes) replicate
on both architectures.** This complicates the exploratory-zoo "more accurate model, larger gap" pattern reported
for AutoBot: Wayformer's accuracy (minADE6 0.928) is comparable to av2_gpu_full's (0.859, different metric
definitions noted in Addendum E-1) and competitive with UniTraj's own reference, yet shows a SMALLER nuScenes gap
than AutoBot while showing a similar-or-larger Waymo gap. The "accuracy predicts gap size" association therefore
does not hold across architectures, only within the single-architecture AutoBot zoo it was observed on; the paper
must soften this claim to be AutoBot-specific, not general, now that it is directly contradicted by a second
architecture's within-dataset comparison.

**Scene-level repair, AV2->Waymo (corrected design, D3/D4, 1000 draws, m up to 300).** Valid at every m (point
estimate 0.981-1.000, all above the 0.88 floor). Same qualitative shape as av2_gpu_full: uninformative (always
infinite) below m~60, reliably finite from m~150 (area 1.62x oracle), falling to 1.35x at m=300. Agent-level C-or-R
stays valid throughout (0.97-0.99) because Waymo's agents-per-scene is small, confirming the mechanism finding
(agent-level failure is specific to large agents-per-scene, i.e. nuScenes) on a second architecture.
AV2->nuScenes scene repair for Wayformer: predictions in progress, outcome to follow in a separate entry.

---

## Addendum E-2 -- Wayformer injection experiments: registered before running -- 2026-10-08
Extends Addendum B-4 (injection design) and Addendum E (Wayformer) to the only architecture-specific gap in the
Wayformer confirmatory battery: Sec.~\ref{sec:inject}'s hz2/lane-dropout/both injections, not yet run on Wayformer
because it reads velocity/acceleration/heading channels that AutoBot ignores (fixed by `recompute_kinematics`,
verified as an exact no-op for AutoBot in the entry above). Registered now, before any Wayformer injection number
is computed: same three conditions (hz2, map, hz2map) and the same composition-reweighting check, same AV2 test
split, same reporting (coverage lost in points and as a share of the real AV2->nuScenes gap G). One caveat specific
to this architecture, fixed now: velocity, acceleration and heading for hz2/jitter are approximated by finite
difference of the (edited) position, since no raw sensor velocity exists for a synthetic trajectory; this is
disclosed in the paper wherever the Wayformer injection numbers are reported, not silently assumed equivalent to
the AutoBot case (which needs no such approximation, since it never reads those channels).

---

## Addendum G -- training-seed sensitivity: scope fixed before training -- written 2026-10-08
Checks whether the headline AV2->nuScenes gap (av2_gpu_full, confirmatory, seed 0, +15.1 points) is sensitive to
the random seed used for training, as opposed to being a property of this particular optimisation run.
**Scope decision, stated honestly.** A full replication (3 seeds x 2 directions x 2 architectures) would need
roughly 2 weeks of additional serial GPU time on this machine's single P2000 and was not undertaken. What is
tractable and highest-value is re-training the single most-quoted model, av2_gpu_full, with one additional seed.
This tests the headline number specifically; it does not test seed-sensitivity of the Waymo transfer, the reverse
direction, or Wayformer, which is disclosed as a residual limitation rather than silently left untested.
**Design (fixed now).** `av2_gpu_full_seed1`: identical recipe, data, gate and epoch budget as av2_gpu_full
(Sec.~setup), seed=1 instead of seed=0, same cache (shape-identical, cache content does not depend on seed). Final
model is the last checkpoint, exactly as for seed 0.
**Analyses run on it once trained (fixed now):** H1 (AV2->nuScenes gap, same-domain control), reported next to
seed 0's. **Success criterion:** the paper's claims are about the SIGN and the ORDER OF MAGNITUDE of the gap, not
its exact value, so the criterion is pre-registered as: seed 1's gap is positive and within a factor of 2 of seed
0's. If it falls outside that band, this is reported as a real finding (seed sensitivity), not explained away.
**Not re-run:** H2-H9, Waymo, scene-repair -- out of scope for a seed-sensitivity check specifically aimed at the
headline number; re-running the entire battery on a second seed is future work, stated as such.

---

## Outcome log -- Addendum E-2, Wayformer injection experiments -- 2026-10-08
`recompute_kinematics` applied (verified no-op for AutoBot, entry above). Results (`results/addB/inject_av2_wayformer.json`):
real gap G = +6.4 pt (matches H1). hz2 (2 Hz history) effect **+8.2 pt, CI [7.1, 9.3], 129% of G**; map (lane
dropout) +1.0 pt (16% of G); both +10.6 pt (166% of G); composition reweighting -1.2 pt (-19% of G, works against
the gap, same direction as AutoBot).
**This differs sharply from AutoBot, where hz2+map together explained only ~27% of a much larger gap (+15.1 pt).**
For Wayformer the measured input difference alone is MORE than sufficient to produce the whole observed gap, with
the CI lower bound (7.1 pt) still exceeding G (6.4 pt) -- not an artifact of noise.
**Caveat that must accompany this number (fixed now, before interpreting it as a clean causal story).**
`recompute_kinematics` derives velocity/acceleration for the injected (2 Hz-resampled, then linearly interpolated)
trajectory by finite difference. A piecewise-linear interpolant has CONSTANT velocity within each 0.2 s segment and
a discontinuous jump in velocity (hence a spiked acceleration) at every original keyframe boundary -- an
artificially jagged motion profile that a real native-2Hz sensor trajectory would not have (real motion is smooth;
only the SAMPLING is coarse). Since Wayformer reads these derived channels and AutoBot does not, this may inflate
the measured hz2 effect for Wayformer specifically, beyond what genuinely-coarser real-world sampling would cause.
**Reading, stated honestly.** The qualitative finding -- input-level differences explain much more of Wayformer's
gap than of AutoBot's -- is consistent with Wayformer's overall smaller, more "explicable" gap (Sec. mechanism) and
is unlikely to be ENTIRELY an artifact, since lane dropout (map, no kinematics derivation involved at all) also
shows a non-trivial, positive effect (16% of G) in the same direction. But the exact hz2/hz2map magnitudes for
Wayformer are not presented as clean causal evidence to the same standard as AutoBot's, and the paper states the
interpolation-smoothness caveat wherever these numbers are quoted.

---

## Addendum H-0 -- H21 (history-length confound) WITHDRAWN: its premise is false for this pipeline -- 2026-10-08
H21 (Addendum D6) was registered on the premise that UniTraj gives Waymo a 1.1 s history padded to 2.1 s, so a
"truncate AV2 history to 1.1 s" injection would reproduce a real AV2->Waymo input difference. The premise came from
Waymo's native format (91 steps, `current_time_index` 10) and was stated in a chat review and in this log WITHOUT
being checked against the pipeline. Checked now, before running anything:
- **Code:** `base_dataset.py:313` sets `ret['current_time_index'] = self.config['past_len'] - 1` (= 20), i.e. UniTraj
  ignores each dataset's native current-time index and windows frames 0..20 as history and 21..80 as future, for
  every dataset. The converted Waymo scenarios do carry native `current_time_index` = 10 (inspected), but it is not
  used.
- **Data:** target agents' valid history frames in the model inputs (6 batches each): AV2 test 21.0/21 (min 21);
  Waymo test 20.6/21 (min 14, per-frame valid share 0.92-1.00, no leading block of padding).
**Consequence.** There is no 1.1 s history on Waymo in this pipeline, so the injection would test a difference that
does not exist; H21 is not run. The "preprocessing artifact" worry therefore applies to nuScenes only (native 2 Hz,
already tested by the hz2 injection), and the AV2->Waymo gap (+23.4/+25.0 pt) has NO measured input-level
explanation: no injection was designed for it because no measured AV2-vs-Waymo input difference was found, which is a
limitation to state, not a result. The review claim "Waymo has a 1.1 s history, padded" (chat, 2026-10-08) and
REVISION_PLAN item 3.1 are corrected here. No paper text or reported number depended on the false premise.

---

## Addendum H -- Method B, protocol-matched label-free recalibration (AV2 -> nuScenes only) -- written 2026-10-08, BEFORE any run
**Idea.** The injection experiments (Addendum B-4/E-2) measure input-level differences between AV2 and nuScenes
(2 Hz history, lane density). Apply the SAME label-free transform T = hz2map (lane keep prob from
`results/addB/inject_params.json`, measured from target INPUTS only) to the source CALIBRATION inputs and take
q_pm = (1-alpha) quantile of the resulting calibration scores. No target label is used, so q_pm is fixed before any
labelled target scene is drawn and Proposition 3's validity (scene rule started from q_pm instead of q_src) is
unchanged. Applies to AV2 -> nuScenes only: no measured AV2-vs-Waymo input difference exists (Addendum H-0), so no
protocol transform is defined for Waymo and none is invented.
**What is measured (fixed now).** (a) Label-free: coverage of q_pm on the nuScenes evaluation pool vs that of
q_src, signed gap, scene-cluster bootstrap CI, per model (av2_gpu_full, av2_wayformer). DESCRIPTIVE: no guarantee is
claimed for q_pm itself. (b) Scene-level rule (Addendum D3/D4 design, 1000 draws, same pools and m grid) with q_pm
as the source threshold; report P[R<=alpha], abstention rate and area vs oracle, and the recommended budget by the
D4 rule next to the q_src result.
**Honest prediction, fixed now, including where I expect it NOT to help.** The rule's budget is set by its power to
reject H_q near the oracle threshold, which does not depend on where the grid is anchored; moving the starting point
therefore cannot be expected to shrink the budget for a tight certificate. It can shrink the budget only if q_pm
over-covers (R(q_pm) well below alpha gives slack, so the certify branch can fire at small m at the price of a
region larger than the oracle). Predictions: (P1) Wayformer, where hz2 alone reproduced 129% of the gap, q_pm closes
at least half of |gap_src| and may overshoot into over-coverage; (P2) AutoBot, where the injections explained ~27%,
q_pm closes less than half of |gap_src|; (P3) the recommended budget (D4 rule) changes by at most 50 scenes for
AutoBot. Anything else is reported as found. Status: EXPLORATORY (new method, not a pre-registered test of a prior
claim); the paper labels it so.
**Caveat carried into the paper.** For Wayformer, T edits position and the derived kinematics are recomputed by finite
difference (Addendum E-2 caveat): q_pm for Wayformer inherits the possibility that the transform is harsher than real
2 Hz sensing.

---

## Outcome log -- Addendum H (Method B, protocol-matched recalibration, AV2 -> nuScenes) -- 2026-10-09
Run exactly as registered (`src/analyze_methodB.py`; transform hz2map on av2cal inputs; CPU inference; `results/rev1/methodB_*.json`).
**(a) Label-free coverage on nuScenes (descriptive; no guarantee for q_pm).**
- av2_gpu_full: q 3.81 m -> 4.69 m; coverage 0.749 [0.713, 0.779] -> 0.811 [0.781, 0.840]; gap +15.1 -> +8.9 pt (**closes 41%**).
- av2_wayformer: q 4.16 m -> 5.71 m; coverage 0.836 [0.807, 0.861] -> 0.923 [0.903, 0.939]; gap +6.4 -> -2.3 pt (**closes 137%, overshoots into over-coverage**).
**(b) Scene-level rule started from q_pm.** Valid at every m on both (P[R<=alpha] 1.000, Wilson lower bound 0.996).
Recommended budget by the D4 rule UNCHANGED: AutoBot m=150 (area 1.91x with q_src, 1.89x with q_pm); Wayformer m=150
(1.77x vs 1.78x). At m=100: 2.71x -> 2.78x (AutoBot), 2.27x -> 2.26x (Wayformer).
**Predictions as registered: P1 (Wayformer closes >= half, may overshoot) CONFIRMED; P2 (AutoBot closes < half) CONFIRMED
(41%); P3 (AutoBot budget changes by <= 50 scenes) CONFIRMED (no change).** The registered caution also held: moving
the starting point does not shorten the certification budget, because the budget is set by the power to reject
near the oracle threshold. Wayformer's over-coverage leaves too little slack (R(q_pm) is ~0.08 vs alpha 0.10) for
the certify branch to fire at small m.
**Two further observations, stated as such.** (1) At m = 80 the abstention rate on AutoBot falls from 57% to 1% with
q_pm but the median area rises (3.07x -> 3.58x): the fixed grid is anchored at q_pm x [4, 0.5], so a larger anchor
raises the top of the grid and lets the first test reject earlier; this is a grid-range effect, not a better
certificate, and the area shows the price. (2) Wayformer's overshoot could partly reflect the finite-difference
kinematics approximation (Addendum E-2); the AutoBot number, which needs no such approximation, is the cleaner
estimate of what protocol matching alone does.
**Reading.** A measured, label-free protocol correction recovers part of the gap (41% for AutoBot) or all of it
(Wayformer), in the conservative direction for the latter, but it is not a substitute for labels and does not reduce
the label budget. It is the restricted-shift-class label-free method that Proposition 1 says a useful label-free
method must be: it works exactly as far as the measured protocol differences explain the shift. EXPLORATORY.

## Outcome log: score ablation (exploratory, not pre-registered as a hypothesis; 2026-10-09)
Script `src/analyze_scores.py`, output `results/rev1/score_ablation.json`. Four alternative scores (top-prob mode, final
displacement, prob-weighted gamma=0.5, time-scaled by source median error fitted on a disjoint half of the cal split) vs the
registered score, same transfers, alpha=0.1, cal = half of the source cal split (so numbers differ slightly from the main
tables). Result: all 24 out-of-domain cells under-cover, by 4 to 26 points. The time-scaled score helps AutoBot AV2->Waymo
(23 -> 9 pts) but hurts AV2->nuScenes (15 -> 24). Conclusion: the coverage loss is not specific to the registered score;
no alternative is uniformly better. Efficiency in metres is reported only as the threshold q (and mean tube radius for the
time-scaled score) in the JSON; region-area comparison across score families was NOT done (regions are not comparable).

## Outcome log: Addendum G (training-seed sensitivity, 2026-10-12)
av2_gpu_full_seed1 trained to completion (60 epochs, last checkpoint, best-ckpt val minADE6 0.875 vs 0.859 for seed 0).
H1 AV2->nuScenes at alpha=0.1: raw gap +20.6 pts (CI 19.2 to 22.5) vs seed 0's +15.1; same-domain gap +0.8 (control ok).
Pre-registered criterion (positive and within a factor 2 of seed 0): MET (ratio 1.36). The gap is larger, not smaller, with
the second seed, so the seed-0 value is not an upper outlier, but the two seeds differ by 5.5 points, more than the seed-0
CI width: seed-to-seed variation is a real component of uncertainty and is stated in the paper. Not re-seeded (disclosed):
reverse direction, Waymo transfers, Wayformer. Injections and Method B were not run on seed 1.

## Addendum I -- Wayformer training-seed sensitivity: scope fixed before training -- written 2026-10-12
Mirrors Addendum G for the second architecture. `av2_wayformer_seed1`: identical recipe, data, cache (C:\p01_cache_wf),
epoch budget (27) and final-model rule (last checkpoint) as av2_wayformer, seed=1 instead of 0. Analyses once trained:
H1 AV2->nuScenes gap and same-domain control, plus the AV2->Waymo gap, reported next to seed 0's (+8.1 pts nuScenes as in
the paper; value read from results/run2 at analysis time). Success criterion, fixed now: seed 1's AV2->nuScenes gap is
positive and within a factor of 2 of seed 0's. The accuracy gate (val minADE6 <= 0.98) is applied to seed 1 exactly as for
seed 0; if seed 1 fails the gate it is reported as a failed training run, not silently replaced. Injections, Method B and the
reverse direction are NOT re-run on seed 1 (disclosed). If the gap falls outside the band it is reported as seed sensitivity.

## Addendum J -- Method A feasibility step 1: can a source-trained loss predictor rank target scenes? -- written 2026-10-13, BEFORE any run
Follow-up-paper idea (Method A): use a label-free predictor of per-scene miss probability as a control variate in an
anytime-valid scene-level test, to reduce the labelled scenes needed. Its value depends entirely on how well a predictor
trained on SOURCE labels ranks TARGET scenes by their true loss. This step measures only that; it does no testing.
**Design (fixed now).** Per agent, label-free features: the 7 `feats_lf` (history-based) + model-output summaries (mean
and max over modes of endpoint distance from the agent's last position, std of mode endpoints, entropy of mode probs,
top-mode probability). Target: miss indicator 1[score > q_src] at alpha=0.1 on the source calibration split, q_src from
that split. Predictor: logistic regression (standardised) and gradient boosting (sklearn defaults, depth<=3, 200 trees),
fit on a random half of source calibration agents, hyperparameters not tuned on target. Scene prediction = mean of agent
predictions in the scene. Evaluation: Spearman rho between predicted and true scene-level miss fraction, and
r^2 = squared Pearson (the variance a control variate could remove), on target scenes: AV2->nuScenes (138 scenes) and
AV2->Waymo test, for av2_gpu_full and av2_wayformer; nuScenes->AV2/Waymo for ns_gpu_full. Same-domain held-out half of
source is the control (expect rho>0 there).
**Predictions (honest).** I expect weak transfer: rho 0.2-0.4 on nuScenes, because the predictor is trained where the
miss rate is 10% and applied where it is 25%+, and nuScenes has only 138 scenes (rho interval about +-0.17).
**Decision rule (fixed now).** Method A is worth developing as a paper only if r^2 >= 0.25 (rho >= 0.5) on at least two
of the AV2->nuScenes / AV2->Waymo cells for both architectures. If not, it is reported as a negative result in the log
and the follow-up is dropped, not rescued by feature engineering after seeing target labels.

## Outcome log: Addendum J (Method A step 1, 2026-10-13) -- NEGATIVE, Method A dropped as designed
Script `src/analyze_methodA_step1.py`, results `results/rev1/methodA_step1.json`. Scene-level r^2 between predicted and true
miss fraction on target scenes: AutoBot AV2->nuScenes 0.08 (logit) / 0.02 (gbm), AV2->Waymo 0.06 / 0.01; Wayformer
AV2->nuScenes 0.11 / 0.16, AV2->Waymo 0.09 / 0.03; nuScenes-trained model to AV2/Waymo 0.00-0.01. Best Spearman 0.50
(Wayformer nuScenes logit, 138 scenes, interval wide). Same-domain held-out control is itself weak (r^2 0.03-0.24), so even
without shift these features barely predict a scene's miss fraction. The predictor also under-predicts the target miss rate
(e.g. 0.11 vs 0.27 on nuScenes), as expected under shift. Decision rule (r^2>=0.25 on >=2 cells for both architectures) NOT
met by any cell. A control variate with r^2<=0.16 could cut labelled-scene variance by at most ~16%, which does not change the
label budget materially. Method A as a variance-reduction paper is dropped; no feature engineering after seeing target labels.
Salvage for the main paper: one sentence in Discussion that scene-level loss is hard to predict from label-free features.

## Outcome log: Addendum I (Wayformer seed 1, 2026-10-19)
av2_wayformer_seed1 trained 27/27 epochs (DONE 2026-10-16; the job crashed and auto-resumed on 2026-10-16 15:04, losing ~8
min). Final (last) checkpoint measured directly with src/validate_ckpt.py: val minADE6 = 0.941 (seed 0: 0.928); gate <= 0.98
PASSED. Results (results/rev1/wf_seed1.json): AV2->nuScenes gap +7.7 pts (CI 6.5-9.0; seed 0 +6.4, CI 5.0-7.9), ratio 1.21;
same-domain gap +0.5 (control ok); AV2->Waymo gap +24.2 (seed 0 +25.0). Registered criterion (positive, within a factor of
2 of seed 0): MET. Wayformer's nuScenes gap stays far below AutoBot's (+15.1 and +20.6 for its two seeds), so the claim that
magnitude does not carry across architectures survives a second seed for both. Seed-to-seed differences: AutoBot 5.5 pts,
Wayformer 1.4 pts (nuScenes). Injections, Method B and the reverse direction were not repeated on seed 1 (disclosed).
(Note: the CIs quoted above are agent-level; scene-clustered 95% intervals are seed 0 3.9-9.1 and seed 1 5.1-10.7 pts.)

## Note (2026-10-19): re-measurement of the Wayformer seed-0 final checkpoint
For reproducibility the final-checkpoint accuracy of both Wayformer runs was re-measured with src/validate_ckpt.py under the
same procedure (2000-scene val subset, seed 0) and stored in results/rev1/wf_gate.json (logs: results/validate_wf_seed*_last.log).
Seed 0 now reads val minADE6 = 0.9319 (the 2026-10-07 entry above recorded 0.9275, a different run of the same procedure; the
subset is not perfectly stable between runs). Seed 1: 0.9408. Gate (<= 0.98) passes for both. The paper now reports the
generated values (0.932 and 0.941). No conclusion changes.

## Correction (2026-10-19): confirmatory accuracies now reported for the registered final model (last checkpoint)
The paper reported AutoBot confirmatory accuracies from the best-validated checkpoint (av2_gpu_full 0.854, ns_gpu_full 1.232)
although the registered final model, used for every coverage result, is the last checkpoint (never validated during training
because the last epoch is not a validation epoch). Measured directly with src/validate_ckpt.py (same split, 2000-scene subset,
seed 0; results/rev1/final_minade.json): av2_gpu_full 0.846, av2_gpu_full_seed1 0.870, ns_gpu_full 1.336 (gate 1.39, so it
still passes, with less headroom than reported: 10% above the 1.21 reference instead of 2%). Wayformer values were already
direct final-checkpoint measurements (0.932, 0.941). No coverage number changes; only the reported accuracies and the
accuracy-related sentences. The exploratory AV2 models' predictions were made from their named checkpoints, whose accuracies
are what the paper reports, so they are unaffected.

## Pre-submission audit corrections (2026-10-19, no new hypothesis)
Found in an end-to-end audit and corrected (generated from result files, no hand-typed values):
- Agent-level C-or-R on small-scene targets: the text said it "held on Waymo and AV2 for every model". False for
  nuScenes->Waymo (88% valid at m=40 falling to 65% at m=300; source threshold misses 27.8% of agents). Now reported,
  and it supports the existing reading that the size of the source miscalibration, not scene size alone, drives failure.
- Intervals: Table I, the Wayformer nuScenes CI and the seed CIs were agent-level although the paper says multi-agent
  targets resample scenes. All now scene-clustered (addB B10). With the wider intervals the two AutoBot seed intervals
  overlap (12-18 vs 17-24); the sentence "seed variation exceeds the sampling interval" was replaced accordingly.
- Accuracy-gap association: holds across training budgets but not between training seeds (seed 1 is less accurate,
  0.870 vs 0.846, yet has the larger gap). Intro, Results, Discussion and Conclusion reworded.
- Audit "false-alarm" on shift-free splits ranges 0-23% across the five confirmatory runs (was reported as <=7% from
  two runs). It tracks the realised threshold's own same-domain shortfall (up to 1.0 pt), i.e. the audit tests the
  realised threshold; stated in the paper, with the margin recommendation.
- Removed from Methods: the pooled/shrinkage/k* recalibration comparison and ACI result (i), never reported in Results.
