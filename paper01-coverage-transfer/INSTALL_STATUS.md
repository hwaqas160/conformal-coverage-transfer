# Paper 01 — Install & Data Status

_Last updated: 2026-09-10_

## ✅ DONE — environment

| Item | Status | Detail |
|---|---|---|
| Python 3.10.11 | ✅ | installed via winget, `py -3.10` |
| venv | ✅ | `F:\CLAUDE\AI1\shared\envs\unitraj` |
| PyTorch | ✅ | `torch 2.0.1+cu118`, `torchvision 0.15.2+cu118` |
| **GPU compute** | ✅ **VERIFIED** | P2000, capability (6,1), real GPU matmul OK |
| numpy | ✅ | pinned `1.24.2` (<2 — torch 2.0.1 requires it) |
| pytorch-lightning | ✅ | `2.0.9` + torchmetrics `1.2.1` |
| scientific stack | ✅ | sklearn 1.3.2, scipy 1.11.4, pandas 2.1.4, matplotlib 3.8.2, pyarrow 14.0.2 |
| wandb / timm | ✅ | wandb 0.16.6 (use `WANDB_MODE=disabled`), timm 0.9.12 |
| metadrive-simulator | ✅ | `0.4.2.3` |
| ScenarioNet | ✅ | editable install, `--no-deps` |
| UniTraj | ✅ | via `.pth` (no `setup.py develop` — avoids CUDA-ext build) |
| setuptools | ✅ | pinned `69.5.1` (newer removed `pkg_resources`, breaks lightning) |
| **Locked env** | ✅ | `code/requirements.lock` — 100 pinned packages |

## ✅ DONE — smoke test

```
cd code/UniTraj/unitraj
WANDB_MODE=disabled python train.py method=autobot
```
Ran 2 epochs on shipped sample data (61 nuScenes scenarios), wrote a checkpoint,
computed `val/brier_fde`. **The full training pipeline works on the P2000.**

Local patches to make this work: see `notes/unitraj_patches.md` (all reversible, `*.orig` kept).

## ✅ DONE — Argoverse 2 Motion Forecasting (raw)

- `data/argoverse2/` — **59 GB, all 3 splits complete and integrity-verified**
  - train 199,908 · val 24,988 · test 24,984 (each dir has scenario.parquet + log_map_archive.json)
- Downloaded with `F:\CLAUDE\AI1\shared\tools\s5cmd.exe --no-sign-request` (no AWS account)
- av2 devkit 0.3.6 installed in the venv; a scenario loads correctly via `scenario_serialization`

## ⏳ IN PROGRESS — AV2 → ScenarioNet conversion

- Script: `src/run_convert_av2.sh <split>` · details + gotchas: `src/convert_av2.md`
- **val** split converting now → `data/av2_scenarionet/val/` (8 workers, ~15 min)
- Then run **test**, then **train** (train is ~200k scenarios — run overnight)
- **Converter dies if the launching shell is killed.** Launch it detached:
  `powershell Start-Process ... -WindowStyle Hidden` (see how it was started, or just
  re-run `src/run_convert_av2.sh` — `--overwrite` makes it restartable, no resume needed
  for val/test; for train, move completed raw dirs aside to resume).

## ❌ CANNOT AUTOMATE — you must do these

| Dataset | Why | Action |
|---|---|---|
| **nuScenes** | License must be accepted by a named account holder | Register at nuscenes.org, then `data/DOWNLOAD.md` |
| **Waymo Open Motion** | Google account + license click-through | waymo.com/open |
| **nuPlan** | Registration | nuscenes.org/nuplan |

Paper 01's minimum viable version = **Argoverse 2 + nuScenes**. AV2 is downloading;
nuScenes is the one gated dataset you need for the first result.

## NEXT — once AV2 finishes

1. Install `nuscenes-devkit` in the venv (for ScenarioNet's nuScenes converter — only when you have nuScenes).
2. Convert AV2 to ScenarioNet format — see `code/scenarionet/documentation/example.rst`
   for the real converter command (do not trust a guessed one).
3. Profile AutoBot on real AV2 data — measure true VRAM and max batch size, record in
   `notes/hardware_limits.md`.
4. Write `notes/falsification.md` (the H1/H2/H3 refutation thresholds) — before any experiment.
5. Start Week 5–7 of `PLAN.md`: train AutoBot on AV2, build `src/conformal.py`.
