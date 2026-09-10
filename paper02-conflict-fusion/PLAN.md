# PAPER 02 — Conflict-Aware Late Fusion
### Which Sensor to Believe When Weather Degrades Them Differently

**Months 4–9 · the core method paper · highest ceiling · your Canada application**

---

## 1. PROBLEM STATEMENT

Camera degrades under poor visibility. LiDAR suffers scattering and attenuation in rain and
snow. Radar is largely immune to both and supplies Doppler velocity directly.

So when the three disagree in a snowstorm, the fusion layer must arbitrate — and current
systems arbitrate using fixed learned weights fitted in good weather. The literature already
reports that **adding a camera under adverse weather sometimes *degrades* performance**. That
is the arbitration failure showing through: the fusion layer cannot tell that one of its
inputs has become noise.

A deployed vehicle cannot retrain its detectors when it starts snowing. It needs a fusion
layer that measures degradation and routes trust accordingly, at runtime.

---

## 2. THE GAP

Reliability-aware fusion exists — **RAF** (arXiv 2607.04587) is the closest 2026 work. It
trains end-to-end, which (a) requires hardware you do not have, and (b) produces a system
nobody can retrofit onto an existing stack.

**Your three differentiators — state all of them explicitly in Related Work:**

1. **Late fusion over frozen, off-the-shelf detectors.** A deployable retrofit, not a new
   stack. This is also what makes it runnable on a P2000.
2. **Per-object degradation-conditioned routing**, not per-frame. A camera can be reliable for
   a near object and useless for a distant one in the same frame.
3. **Formal risk guarantees** on the fused output, carried over from Paper 01's machinery.

Cite RAF first and state your difference precisely. Reviewers will find it regardless; better
that you framed the comparison than that they did.

---

## 3. HYPOTHESES

Write to `notes/falsification.md` before experimenting.

- **H1 (harm exists).** There is a measurable set of cases where adding a degraded modality
  reduces detection quality versus omitting it. *Refuted if* the harm rate is under 2%.
- **H2 (predictable).** A degradation-conditioned head predicts those harm cases better than
  chance and better than a per-frame weather classifier. *Refuted if* per-object routing does
  not beat per-frame routing.
- **H3 (guarantee transfers).** Risk control calibrated on clear weather holds under snow.
  *Refuted if* empirical coverage collapses as in Paper 01 — **which is itself a publishable
  link between the two papers.**

---

## 4. CONTRIBUTIONS TO CLAIM

1. A per-object, degradation-conditioned arbitration head that sits on top of **any** frozen
   detectors.
2. Risk-controlled fusion: the fused output carries a coverage guarantee, and residual
   cross-sensor conflict inflates the nonconformity score.
3. Cross-condition external validation — calibrate in clear weather, deploy in snow.
4. Quantification of the **harm case**: how often does adding a modality make things worse,
   and can it be predicted in advance.

---

## 5. DATASETS

| Dataset | Role | Detail | Access |
|---|---|---|---|
| **Boreas** | Primary | 350+ km, repeated route over a full year. 128-ch Velodyne Alpha-Prime LiDAR, 360° Navtech CIR304-H scanning radar, 5 MP FLIR Blackfly S camera, cm-accurate ground truth. IJRR published. | Open AWS S3, **no account needed** |
| **CADC** | Snow validation | Canadian Adverse Driving Conditions, ~7k annotated frames of snowy driving | Waterloo, registration |
| **nuScenes** | Clear-weather calibration | Reuse from Paper 01 — no extra download | Already have it |
| SeeingThroughFog / DENSE | Optional fog | Adds a third degradation type | Registration |

See `data/DOWNLOAD.md`. Boreas commands are verified from the pyboreas README.

---

## 6. STRATEGIC NOTE — this is your Canada application, embedded in a paper

**Boreas was built by Barfoot's ASRL at UTIAS. CADC came out of Waterloo.**

If you publish a strong result on both, your application letter says *"I extended your
benchmark"* rather than *"I admire your work."* That is the entire reason these datasets were
chosen over nuScenes-only. Do not substitute them for convenience.

---

## 7. CODE

- **`code/pyboreas/`** — official UTIAS ASRL devkit. Cloned.
  - `BoreasDataset` loader, calibration handling, projection utilities
  - Download instructions verified in its README
- **Detectors:** do **not** train any. Use published pretrained checkpoints, run them in
  inference mode once, and cache every detection and feature to disk.
- **Your code:** `src/arbitrate.py` — a small GNN or MLP over cached detections. A few million
  parameters at most.

---

## 8. THE CRITICAL COMPUTE PATTERN

> **Run frozen detectors ONCE, offline. Cache everything. Never run them again.**

This is what makes the paper possible on a 5 GB Pascal card.

```
Week 1 of P02:  cache_detections.py  ->  runs for ~1-2 weeks in background
                                          outputs: per-frame detections, features,
                                          confidence, per-object range/visibility
Weeks after:    train arbitration head on cached tensors  ->  minutes per epoch
```

**Budget two weeks for the inference pass, not two days.** Start it in month 4 while Paper 01
is under review — this is the single most important scheduling decision in the programme.

**If you find yourself wanting to train a detector, the paper has drifted out of scope.**

---

## 9. WEEK-BY-WEEK PLAN

### Weeks 1–2 (programme months 4–5) · Data + offline inference
- [ ] Start Boreas download (see `data/DOWNLOAD.md`). It is large — start early.
- [ ] Install pyboreas, verify `BoreasDataset` loads a sequence.
- [ ] Select pretrained detectors for camera / LiDAR / radar. Record exact checkpoints.
- [ ] Write and launch `src/cache_detections.py`. **Let it run in background for 2 weeks.**

### Weeks 3–4 · Degradation characterisation
- [ ] Build `src/degradation.py`: per-object features — range, point density on object,
      image contrast/blur in the object crop, radar RCS and Doppler consistency.
- [ ] Quantify **H1**: how often does 3-sensor fusion underperform a 2-sensor subset?
- **GATE:** harm rate measured. If under 2%, the paper's premise is weak — tell me and pivot.

### Weeks 5–7 · The arbitration head
- [ ] `src/arbitrate.py`: per-object trust weights conditioned on degradation features.
- [ ] Baselines: (a) fixed-weight late fusion, (b) per-frame weather classifier routing,
      (c) confidence-only routing.
- **GATE:** H2 answered — per-object beats per-frame.

### Weeks 8–10 · Risk control + cross-condition validation
- [ ] Port `conformal.py` from Paper 01. Add cross-sensor conflict to the nonconformity score.
- [ ] Calibrate on clear-weather Boreas sequences, deploy on snow sequences and CADC.
- [ ] Report empirical vs nominal coverage — the direct link to Paper 01.
- **GATE:** H3 answered.

### Weeks 11–14 · Write and submit
- [ ] Related Work: position against RAF precisely, in writing, before you finalise claims.
- [ ] arXiv on submission day.

---

## 10. EXPERIMENTS

| # | Experiment | Proves |
|---|---|---|
| 1 | Harm-rate quantification across weather conditions | H1 — the motivating measurement |
| 2 | Per-object vs per-frame vs fixed-weight routing | H2 — the core claim |
| 3 | Ablation over degradation features | Which signals actually carry the routing decision |
| 4 | Clear → snow risk-control transfer | H3, and the link to Paper 01 |
| 5 | Boreas → CADC cross-dataset transfer | Generalises beyond one collection |
| 6 | Comparison against RAF-style end-to-end (cited numbers) | Honest positioning |
| 7 | Per-modality dropout stress test | Graceful degradation when a sensor fails entirely |
| 8 | Qualitative cases incl. one failure | Reviewers at these venues distrust papers without a failure case |

---

## 11. TARGET JOURNALS

| Priority | Venue | IF | Quartile | Cost |
|---|---|---|---|---|
| 1 | **Information Fusion** (Elsevier) | 15.5 | Q1 | Free on subscription route; APC $4,360 only if you choose OA |
| 2 | **IEEE T-IV** | 14.3 | Q1 | Free traditional route |
| 3 | IEEE T-ITS | 9.1 | Q1 | Free traditional route |
| 4 | IEEE Sensors Journal | — | Q1/Q2 | Free traditional route |

Information Fusion is an exact scope match and the highest impact available to you. It is also
slow and demanding — budget for a long review.

---

## 12. TARGET LABS

| Lab | Institution | Why |
|---|---|---|
| [ASRL](https://asrl.utias.utoronto.ca/~tdb/) | UToronto UTIAS 🇨🇦 | Barfoot's group **built Boreas**. Publishing on their benchmark is the strongest possible introduction. |
| [WAVELab](http://wavelab.uwaterloo.ca/) | Waterloo 🇨🇦 | AV state estimation and perception; Waterloo produced CADC |
| [TRAILab](https://www.trailab.utias.utoronto.ca/) | UToronto UTIAS 🇨🇦 | Waslander: 3D detection and tracking |
| [AMRL](https://amrl.cs.utexas.edu/) | UT Austin 🇺🇸 | Biswas: introspective perception — robots that know when they are failing |

---

## 13. RISKS

| Risk | Severity | Mitigation |
|---|---|---|
| Offline inference pass is the one place the P2000 genuinely hurts | **High** | Start it month 4, run in background, budget 2 weeks |
| RAF and successors occupy adjacent ground | **High** | Settle positioning in writing *before* building |
| Boreas download size / time | Medium | Start early; use `--include` filters to pull only needed sensors |
| Pretrained detectors may not exist for Boreas' exact radar format | Medium | Navtech scanning radar is unusual — check pyboreas examples first; fall back to LiDAR+camera and treat radar as the robustness probe |
| Harm rate turns out negligible | Medium | Gate at week 4 catches this before you build the method |

---

## 14. REFERENCE PAPERS (downloaded in `papers/`)

| File | Why it matters |
|---|---|
| `Boreas_dataset_IJRR.pdf` | Your primary dataset, by your target lab. **Read first.** |
| `RAF_reliability_aware_fusion.pdf` | **Nearest competitor.** Position against it precisely. |
| `4DRadar_cooperative_adverse_weather.pdf` | 2026 radar-under-weather work |
| `LRC_WeatherNet.pdf` | LiDAR/radar/camera weather classification — your per-frame baseline |
| `FogDrive_graded_fog.pdf` | Graded fog degradation, synthetic |
| `SafetyMonitoring_ML_perception_survey.pdf` | Survey — Related Work scaffolding |
