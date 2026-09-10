# PAPER 01 — Does Uncertainty Transfer?
### Conformal Coverage Collapse in Cross-Dataset Trajectory Prediction

**Months 1–5 · lowest risk · builds all shared infrastructure · START HERE**

---

## 1. PROBLEM STATEMENT

Conformal prediction (CP) is sold to the autonomous-driving community as providing
**distribution-free, finite-sample coverage guarantees**: choose alpha = 0.1, and the true
future trajectory falls inside your prediction region 90% of the time.

That guarantee rests on one assumption: **exchangeability** between the calibration set and
the test set. Cross-dataset deployment violates it outright. Argoverse 2 is Pittsburgh and
Miami; nuScenes is Boston and Singapore; they differ in sampling rate, map topology, agent
density, sensor stack and driving culture.

Everybody knows *accuracy* degrades under domain shift — UniTraj (ECCV 2024) measured exactly
that, in ADE/FDE. **Nobody has published whether the guarantee itself survives.**

If a 90% guarantee silently becomes 61% coverage the moment the vehicle crosses a domain
boundary, every downstream safety argument built on it is void. That is the question this
paper answers.

---

## 2. THE GAP (why this is publishable)

Two literatures run in parallel and have not met:

| Literature | What it reports | What it never reports |
|---|---|---|
| Cross-dataset generalisation (UniTraj; goal-based transfer; latent scene embeddings) | ADE / FDE degradation | Whether coverage guarantees hold |
| Conformal for trajectories (CUQDS; robust CP under shift; scenario-aware UQ) | Coverage under *online* shift **with ground-truth feedback** | Zero-label cross-dataset transfer |

**The untouched cell:** calibrate on dataset A, deploy on dataset B, receive **no labels back**.
Does 90% coverage hold? This is the realistic deployment setting and it is unstudied.

---

## 3. HYPOTHESES

Write these to `notes/falsification.md` **before** running any experiment.

- **H1 (collapse).** Split-conformal coverage calibrated on dataset A drops materially below
  nominal on dataset B. *Refuted if* empirical coverage stays within ±2 points of nominal
  across all transfer pairs.
- **H2 (attribution).** Coverage loss is predominantly attributable to a small number of
  separable shift factors (sampling rate, agent density, map topology), not to an
  irreducible dataset gestalt. *Refuted if* controlling for all measured factors explains
  less than 25% of the gap.
- **H3 (repair).** A label-free recalibration using only unlabelled target scenes restores
  coverage to within 3 points of nominal, without target ground truth. *Refuted if* it fails
  to beat naive uncorrected CP.

**H1 producing a negative result is still a paper.** "Your safety guarantee evaporates at the
dataset boundary" is a finding a safety-critical field must hear. That is exactly why this is
the low-risk first paper — it cannot fail into nothing.

---

## 4. CONTRIBUTIONS TO CLAIM

1. First systematic measurement of conformal coverage under **cross-dataset** transfer — a
   full calibrate-here / deploy-there matrix over up to 4 datasets in one unified pipeline.
2. **Attribution** of coverage failure to separable shift factors. This is the part that
   gets cited.
3. A **label-free recalibration** method restoring coverage without target labels.
4. Released benchmark and code so that coverage transfer becomes a standard reported number.

---

## 5. DATASETS

| Dataset | Role | Approx size | Access |
|---|---|---|---|
| **nuScenes** | Start here — sample already in repo | ~40 GB | Free registration, nuscenes.org |
| **Argoverse 2** Motion Forecasting | Primary calibration source | ~200 GB | Open AWS S3, no account needed |
| **Waymo Open Motion** | Third domain | Large | Free registration, Google account |
| **nuPlan** | Fourth domain | Very large | Free registration, nuscenes.org |

See `data/DOWNLOAD.md` for exact commands.

**Minimum viable paper = Argoverse 2 + nuScenes only.** Add Waymo and nuPlan if disk and time
allow. Do not block the paper on having all four.

---

## 6. CODE (already cloned into `code/`)

- **`code/UniTraj/`** — ECCV 2024. Unified training and evaluation across all four datasets.
  - Models present: `autobot`, `mtr`, `wayformer`, `smart`, `emp`, `fmae`
  - Entry points: `unitraj/train.py`, `unitraj/evaluation.py`
  - Configs: `unitraj/configs/config.yaml`, `unitraj/configs/method/*.yaml`
  - Real command from its README: `python train.py method=autobot`
  - Ships sample nuScenes data at `unitraj/data_samples/nuscenes`
- **`code/scenarionet/`** — NeurIPS 2023. The data conversion layer. Install this first.

**Model choice is dictated by your GPU:**

| Model | Params | P2000 verdict |
|---|---|---|
| **AutoBot** | **1.5 M** | Primary model. Comfortable. |
| Wayformer | 16.5 M | Secondary, reduced batch |
| MTR | 60.1 M | Burst to Kaggle/Colab only |

---

## 7. WEEK-BY-WEEK PLAN

### Weeks 1–2 · Environment and smoke test
- [ ] Follow `shared/envs/SETUP.md`. Verify `torch.cuda.get_device_capability(0) == (6,1)`.
- [ ] Set `debug: False` and `load_num_workers: 4` in `unitraj/configs/config.yaml`.
- [ ] Set `train_batch_size: 16` in `unitraj/configs/method/autobot.yaml` — the default of
      128 **will** OOM on 5 GB.
- [ ] Run `python train.py method=autobot` on the shipped sample data.
- [ ] Record the maximum batch size that fits → `notes/hardware_limits.md`.
- **GATE:** AutoBot trains on GPU without OOM. Nothing proceeds until this passes.

### Weeks 3–4 · Data pipeline
- [ ] Register for nuScenes and Waymo. Start the Argoverse 2 S3 pull (no account needed).
- [ ] Convert each dataset to ScenarioNet unified format.
- [ ] Build `src/shift_factors.py`: for every scene compute sampling rate, agent count,
      map-element density, lane-graph complexity, ego speed distribution.
- **GATE:** two datasets fully converted and loadable through UniTraj.

### Weeks 5–7 · Train and calibrate
- [ ] Train AutoBot on Argoverse 2, holding out a dedicated calibration split.
- [ ] Implement `src/conformal.py`: split CP over trajectory prediction regions.
      Nonconformity score = normalised displacement error at each horizon step.
- [ ] Verify **in-domain** coverage hits nominal. If it fails here, your CP implementation is
      wrong — not the world.
- **GATE:** in-domain empirical coverage within ±1 point of nominal at alpha ∈ {0.05, 0.1, 0.2}.

### Weeks 8–10 · The core experiment
- [ ] **Transfer matrix.** For every ordered pair (A calibrate, B deploy), report empirical
      coverage vs nominal, plus mean region size.
- [ ] Repeat across 3 seeds and 3 alpha levels.
- [ ] Repeat with AutoBot and Wayformer to show it is not model-specific.
- **GATE:** H1 answered. This is your headline figure.

### Weeks 11–13 · Attribution and repair
- [ ] Regress the coverage gap on the shift factors computed in week 3.
- [ ] Implement label-free recalibration in `src/recalibrate.py`. Candidate approaches:
      importance-weighted CP with a density ratio estimated from *unlabelled* target features;
      or a shift-aware nonconformity score conditioned on measured scene statistics.
- [ ] Show it restores coverage without target labels.
- **GATE:** H2 and H3 answered.

### Weeks 14–17 · Write and submit
- [ ] Draft. Related Work can be written **now** — the 8 PDFs in `papers/` are the skeleton.
- [ ] Every table regenerable by one script in `scripts/`.
- [ ] **Post to arXiv the same day you submit.** That link is what you send to supervisors.
- **GATE:** submitted and on arXiv.

---

## 8. EXPERIMENTS

| # | Experiment | Proves |
|---|---|---|
| 1 | In-domain coverage sanity check | Your CP implementation is correct |
| 2 | Cross-dataset transfer matrix | H1 — the headline |
| 3 | Coverage vs alpha (0.05 / 0.1 / 0.2) | Not an artefact of one confidence level |
| 4 | Two model architectures | Not a quirk of AutoBot |
| 5 | Shift-factor attribution regression | H2 — the citable part |
| 6 | Label-free recalibration | H3 — turns critique into contribution |
| 7 | Region size / efficiency analysis | Repair does not buy coverage by being uselessly wide |
| 8 | Ablation over nonconformity score design | Result is not score-specific |

**Metrics:** empirical coverage vs nominal; mean and median region area; conditional coverage
by prediction horizon; ADE/FDE for reference.

**Statistics:** 3 seeds minimum, reported mean ± SD. Bootstrap confidence intervals
**clustered at scenario level**, not sample level. Paired tests when comparing on identical
scenarios. You are writing a paper arguing other people's uncertainty estimates are
untrustworthy — your own statistics must be unimpeachable.

---

## 9. TARGET JOURNALS

| Priority | Venue | IF | Quartile | Cost |
|---|---|---|---|---|
| 1 | **IEEE RA-L** | 5.3 | Q2 | Free. 6 pages, max 8, $175 per extra page |
| 2 | **IEEE T-IV** | 14.3 | Q1 | Free on traditional route |
| 3 | IEEE T-ITS | 9.1 | Q1 | Free on traditional route |

RA-L if you want speed plus an ICRA/IROS talk before application season. T-IV if you want the
impact factor. **Do not tick "open access" at submission** on IEEE journals — that commits you
to an APC on acceptance.

---

## 10. TARGET LABS

| Lab | Institution | Why this paper speaks to them |
|---|---|---|
| [xLAB — Safe Autonomous Systems](https://xlab.upenn.edu/) | UPenn | They effectively own [conformal prediction for robotics](https://xlab.upenn.edu/conformal-prediction-robotics/). Best match in the world for this paper. |
| [Stanford ASL](https://stanfordasl.github.io/) | Stanford | Pavone: risk-sensitive planning, uncertainty-aware autonomy |
| [TRAILab](https://www.trailab.utias.utoronto.ca/) | UToronto UTIAS | Waslander: motion prediction for self-driving |

**Outreach protocol:** email in months 6–9 with the arXiv link. Name the specific paper of
theirs you built on. One sentence on what you would work on with them. Never mass-email —
these groups get dozens of generic enquiries weekly and answer the ones showing evidence of
reading.

---

## 11. RISKS

| Risk | Severity | Mitigation |
|---|---|---|
| Someone publishes the same measurement first — the ingredients are all public | **High** | Move fast, arXiv early, lean on attribution and repair which are harder to replicate quickly |
| `natten` / `av2` install failures on Windows | Medium | AutoBot needs neither. Comment them out of requirements.txt. |
| Disk pressure from four datasets | Medium | 1.5 TB is enough for AV2 + nuScenes. Start with two. |
| A CP implementation bug masquerading as a finding | Medium | Experiment 1 exists precisely to catch this. Never skip it. |
| Batch size 128 default causes immediate OOM | Low | Already documented. Set to 16. |

---

## 12. REFERENCE PAPERS (downloaded in `papers/`)

| File | Why it matters |
|---|---|
| `UniTraj_ECCV2024.pdf` | The framework you build on. **Read first.** |
| `CUQDS_conformal_under_shift.pdf` | Nearest competitor — online shift *with* feedback. Position against this precisely. |
| `RobustCP_environments_shift.pdf` | Robust CP under distribution shift |
| `GoalBased_crossdataset.pdf` | Cross-dataset generalisation, accuracy only |
| `Transferability_latent_scene.pdf` | Transferability analysis |
| `ScenarioAware_UQ_guarantees.pdf` | Scenario-aware UQ with statistical guarantees |
| `ScenarioNet_NeurIPS2023.pdf` | The data layer |
| `TrajPred_survey_limits.pdf` | Survey — scaffolding for Related Work |
