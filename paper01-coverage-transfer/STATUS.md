# Paper 01 — Live Status

_Updated 2026-09-10, mid-session_

## Running now (detached, survive session end)

| Job | PID | State | ETA |
|---|---|---|---|
| AutoBot training (av2 val-split, 20 ep) | 45512 | epoch 0/20, ~16 min/epoch | ~6 h |
| AV2 train-split conversion | (Start-Process) | ~50% (100k/200k scenarios) | ~1–2 h |

Training output: `results/train_av2_valsplit_v1.{out,err}`, ckpts in `results/ckpts/av2_valsplit_v1/`
Conversion output: `data/av2_scenarionet/train/`, log `data/convert_train.{log,err}`

## Done this session

- **Environment** — Python 3.10 venv, torch 2.0.1+cu118, **P2000 GPU compute verified**.
  Locked in `code/requirements.lock` (121 pkgs). Full recipe in `../shared/envs/SETUP.md`.
- **Data** — AV2 Motion Forecasting raw (59 GB, 249,880 scenarios) downloaded + integrity-checked.
  - `av2_scenarionet/val` converted: 24,988 scenarios, 0 errors, loads through UniTraj (22,979 valid samples)
  - `av2_scenarionet/train` converting now
  - AV2 **test** split is unusable (challenge holdout, no GT) → internal test carved from train/val
- **Scene-level splits** — `av2_splits/val/{train,cal,test}` = 14,990 / 4,971 / 5,027,
  deterministic by scenario-ID hash (`src/split_db.py`).
- **Research code** (all committed, all tested):
  - `src/conformal.py` — split conformal + 3 recalibration methods (H3). Synthetic: coverage
    matches nominal within 0.4 pt.
  - `src/shift_factors.py` — 7 pre-registered shift factors from a UniTraj batch.
  - `src/unitraj_bridge.py` — load ckpt, build loader over any DB, inference → numpy. Verified.
  - `src/predict.py` — dump (preds, GT, scores, feats) npz per (ckpt, dataset).
  - `src/coverage_matrix.py` — calibrate-A/deploy-B transfer matrix + H2 + H3 comparison.
  - `src/train_autobot.py` — P2000-tuned training. Benchmarked: batch 32 = 16.3 samp/s.
  - `src/tests/test_pipeline_synthetic.py` — end-to-end analysis test on fabricated data. **PASSES.**
- **Falsification note** — `notes/falsification.md`: H1/H2/H3 with numeric refutation thresholds,
  written before any real experiment.

## Blocked on YOU

- **nuScenes** — register at nuscenes.org, accept licence, download. This is the cross-domain
  test set for H1. Cannot be automated. Then: `pip install nuscenes-devkit` in the venv,
  `python -m scenarionet.convert_nuscenes ...` (see `src/convert_av2.md` gotchas — same apply).
- Optionally Waymo Open Motion + nuPlan for extra domains.

## Next (once training + train-conversion finish)

1. Check `av2_valsplit_v1` best minADE6. Kill condition: must beat ~0.85 (≤15% off UniTraj's
   ~0.73). If far worse → more epochs / full train data / investigate.
2. `predict.py` on `av2_splits/val/{cal,test}` → `results/preds/av2_from_av2.npz` (calib+test halves)
3. `predict.py` on nuScenes (when available) → `results/preds/ns_from_av2.npz`
4. `coverage_matrix.py --mode coverage` → first H1 number (Δ_in vs Δ_cross)
5. Retrain on full AV2 train (199k) for the paper's headline model — likely needs Kaggle/Colab
   for a reasonable epoch count, or accept ~30 epochs over ~2 days on the P2000.
