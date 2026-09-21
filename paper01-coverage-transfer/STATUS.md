# Paper 01 — Live Status

_Updated 2026-09-21_

## Headline: first real cross-dataset measurement exists (preliminary — one pair, weak model)

Source model: AutoBot trained on Argoverse 2 (`av2_cpu_v1`). Target: nuScenes val (9,041 samples).
Full numbers: `results/pair_av2_cpu_v1__ns.json`; interpretation in `notes/falsification.md`.

| | Result (alpha = 0.10, nominal 90%) |
|---|---|
| In-domain / same-domain held-out coverage | 89.5% / 90.2% — split-conformal is implemented correctly |
| **nuScenes coverage (zero labels, no recalibration)** | **86.6%** — gap **+3.4 pts** (95% CI +2.7..+4.1), unsafe direction, present at all 3 alphas |
| Pre-registered H1 prediction (>= 5 pts) | **Not met** — gap is real but smaller than predicted |
| H2 (7 shift factors explain the gap) | **Refuted** — adj R^2 = 0.15 (< 0.20), reweighting removed -57% |
| H3 (label-free repair) | **Not supported** — no method beats doing nothing; importance weighting made it worse |

What this supports so far: "the conformal guarantee measurably under-covers across datasets, and standard
covariate-shift reweighting does not repair it." What it does **not** yet support: a large collapse, a
working fix, or any claim beyond one AV2->nuScenes pair.

## Corrections to earlier claims (be aware when reading older files)

- **The "AutoBot AV2 minADE6 ~0.73" yardstick was unverified** (written from memory; not in the UniTraj
  paper). The verifiable reference is nuScenes minADE5: AutoBot-UniTraj 1.26, vanilla 1.37 — trained on
  8xA100, batch 128. Our zero-shot AV2->nuScenes minADE5 is 1.99 (not apples-to-apples). The 0.85 kill
  threshold is unfounded; see the erratum in `notes/falsification.md`.
- The earlier explanation "the model only saw 13,770 scenes" was incomplete: the 60,000-scene run scored the
  same (val minADE6 1.349 vs 1.347). Train ADE was still falling (1.30 at epoch 9) and the LR schedule
  never decayed within 10 epochs — the model is **under-trained (steps/epochs), not data-starved**.

## Jobs

| Job | State |
|---|---|
| AV2 train conversion (199,908) | done |
| nuScenes val conversion | done — 9,041 scenarios (`data/nuscenes_scenarionet/val`) |
| `av2_cpu_v1` training (60k scenes, 10 ep, CPU) | done — best `epoch08-minADE1.349.ckpt`. Attempt 1 crashed at ~3h; the retry wrapper resumed from `last.ckpt` and finished (exit 0) — the hardening worked in practice |
| Watcher `P01_Watch` (5 min) | still running, harmless; logs to `logs/heartbeat.log` |

## Known weaknesses to fix before this can be a paper

1. **One source->target pair.** Need at least: nuScenes->AV2 (reverse), and ideally Waymo/nuPlan.
2. **Weak base model** (AV2 held-out miss rate ~52%). A reviewer will ask for a competent forecaster.
   Needs a proper long run (100+ epochs, batch >= 128, LR decay) — realistically Kaggle/Colab GPU, not the
   contended P2000 or CPU (~2.5 h/epoch).
3. **Sampling-rate confound — tested, does NOT explain the gap (partially).** nuScenes is 2 Hz native, interpolated
   to 10 Hz; AV2 is 10 Hz native. Scoring only at native 2 Hz / 1 Hz time steps gives gaps of +3.5 / +3.6 pts
   (vs +3.4 at 10 Hz) — `results/confound_check_sampling_rate.json`. The ground-truth-smoothness artifact is
   therefore not the cause. NOT tested: the model's *input* history is also interpolated for nuScenes, and
   annotation/tracking-noise differences remain possible.
4. Single training seed (pre-registration asked for 3).

## Blocked on YOU

Nothing is blocked on you for the next analysis steps. Decision needed on direction — see chat.
