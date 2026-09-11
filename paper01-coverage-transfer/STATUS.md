# Paper 01 — Live Status

_Updated 2026-09-11, afternoon_

## Running now (Task Scheduler `P01_TrainFullV1` + `P01_Watch`)

| Job | State | Note |
|---|---|---|
| **AutoBot training on FULL AV2 train (199,908 scenes)** | preprocessing/caching, not yet in the training loop | 15 epochs, batch 32, exp=`av2_full_v1` |
| Heartbeat watcher | every 5 min | `logs/heartbeat.log`; will auto-run predict+coverage when this finishes |

⚠️ **GPU contention**: the P2000 (5 GB) is shared with your other project
(`SupChain2/04_run_ablation_sweep.py`), which was using ~4.9 GB / 100% util when this
launched. My process did acquire a CUDA context and start preprocessing without an
immediate OOM, but if it crashes with `CUDA out of memory` once actual training batches
start, that's why — the two jobs are fighting over 5 GB. Check
`results/train_av2_full_v1.err` for that error if progress stalls. I did not touch your
other job; let me know how you want to prioritize GPU time if this becomes a problem.

## Completed: pipeline-validation training run (`av2_valsplit_v1`)

Finished 20 epochs on the 13,770-scene val-split subset. **Best checkpoint: minADE6=1.347
— misses the ~0.85 kill condition** (≤15% off UniTraj's reported AV2 AutoBot minADE6 of
~0.73). Not a pipeline bug: this model only ever saw 13,770 scenes, versus the ~200k
UniTraj trained on. That's why `av2_full_v1` (above) was launched immediately once the
full AV2 train conversion finished.

Its outputs (checkpoints, predictions, coverage_transfer.json) are preserved in
`results/archive_valsplit_v1/` rather than deleted, in case the smaller model is ever
useful as a fast-iteration baseline.

**Important caveat on that archived coverage result**: the auto-triggered coverage check
compared three files (`av2`, `av2cal`, `av2test`) that are all the SAME underlying
distribution (different halves of AV2's val split) — so its "cross" numbers
(Δ≈-0.0016) are **not a real cross-dataset result**, just confirmation that split
conformal prediction holds coverage on real (non-synthetic) model output. The actual H1
cross-dataset test still needs nuScenes.

## Done this session (cumulative)

- **Environment** — Python 3.10 venv, torch 2.0.1+cu118, **P2000 GPU compute verified**.
  Locked in `code/requirements.lock` (129 pkgs).
- **Data** — AV2 Motion Forecasting raw (59 GB, 249,880 scenarios), fully downloaded +
  integrity-checked. Both `val` (24,988) and now `train` (199,908, +errors=0) converted
  to ScenarioNet format. AV2 **test** split confirmed unusable (challenge holdout, no GT).
- **Scene-level splits** — `av2_splits/val/{train,cal,test}` = 14,990 / 4,971 / 5,027,
  deterministic by scenario-ID hash (`src/split_db.py`), used as the calibration/test
  reference regardless of which model is being evaluated (AV2's train/val split is
  disjoint at the source, so this never leaks into `av2_full_v1`'s training data).
- **Research code** (all committed, all tested end-to-end on synthetic data with a
  realistic covariate shift):
  - `src/conformal.py` — split conformal + 3 recalibration methods (H3)
  - `src/shift_factors.py` — 7 pre-registered shift factors
  - `src/attribution.py` — H2 regression + reweighting removal-fraction
  - `src/unitraj_bridge.py`, `src/predict.py`, `src/coverage_matrix.py`
  - `src/train_autobot.py` — P2000-tuned training
  - `src/tests/test_pipeline_synthetic.py` — **PASSES**: Δ_in≈0, Δ_cross detected,
    weighted recalibration wins clearly, attribution adj_R²=0.84
- **nuScenes path ready** — `nuscenes-devkit` installed, `src/run_convert_nuscenes.sh`
  written and its converter's `-h` verified to run. Only the data download is pending.
- **Job infrastructure** — learned the hard way that `Start-Process -WindowStyle Hidden`
  does NOT survive the CLI host session ending (both jobs died overnight with no error,
  sleep was ruled out via `powercfg`). Switched to Windows Task Scheduler, verified
  running in a separate session. `src/watch.py` + `P01_Watch` logs a heartbeat every 5
  min and auto-runs the next step on completion — see `run/README.md`.

## Blocked on YOU

- **nuScenes** — register at nuscenes.org, accept licence, download to `data/nuscenes/`.
  This is the cross-domain test set for the actual H1 result. Everything downstream
  (`src/run_convert_nuscenes.sh`, `predict.py`, `coverage_matrix.py`) is ready to go the
  moment the data lands. See `data/DOWNLOAD.md`.

## Next

1. Wait for `av2_full_v1` to finish (or crash — see GPU contention note above).
2. `watch.py` will auto-run predict.py on `av2_splits/val/{cal,test}` and a fresh
   in-domain coverage check the moment it's done — no manual step needed.
3. Once you have nuScenes: convert it, run `predict.py --ckpt <av2_full_v1 best> --dataset
   ns --tag ns_from_av2_full_v1`, drop the npz into `results/preds/`, rerun
   `coverage_matrix.py`. **That produces the actual H1 number.**
