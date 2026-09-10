# Environment Setup — AS BUILT

This records what was actually installed and verified on this machine, not a generic recipe.

## Machine

```
GPU  : Quadro P2000, 5120 MiB, Pascal GP106, compute capability (6, 1), driver 556.39
CPU  : i9-10980XE, 18C/36T
RAM  : 64 GB (~47 GB free)
DISK : F: ~1.4 TB free
OS   : Windows 10 Pro 19045
```

## Python

- System Python 3.12.10 — left untouched (bare `python` hits the WindowsApps stub; use `py`).
- **Python 3.10.11 installed** via `winget install Python.Python.3.10 --scope user`.
- **venv at `F:\CLAUDE\AI1\shared\envs\unitraj`** (Python 3.10.11).

Activate it:
```
F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\activate         # cmd / powershell
source F:/CLAUDE/AI1/shared/envs/unitraj/Scripts/activate  # git bash
```
Or call the interpreter directly:
```
F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe
```

## PyTorch — VERIFIED WORKING ON THE P2000

```
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4     # MUST downgrade — torch 2.0.1 crashes on numpy 2.x
```

Verified:
```
torch 2.0.1+cu118 | cuda build 11.8 | torchvision 0.15.2+cu118 | numpy 1.26.4
torch.cuda.is_available()        -> True
torch.cuda.get_device_name(0)    -> Quadro P2000
torch.cuda.get_device_capability -> (6, 1)
GPU matmul 1000x1000             -> OK
```

**This is the milestone the whole research programme depended on. It passed.**

## The numpy trap

`torch 2.0.1` was built against numpy 1.x. Installing anything that pulls numpy 2.x
(`"numpy.core.multiarray failed to import"`) breaks torch AND torchvision.
`paper01-coverage-transfer/code/constraints.txt` pins `numpy<2` — pass it to every install:
```
pip install -c F:\CLAUDE\AI1\paper01-coverage-transfer\code\constraints.txt <packages>
```

## The pip resolver trap

Installing UniTraj's `requirements.txt` unpinned + a `numpy<2` constraint sends pip's
resolver into multi-GB backtracking (it downloads every historical version of `wandb`,
`pytorch-lightning`, etc. looking for a consistent set). **Install in small pinned batches
instead.** The batches that worked:

```
# Batch A — core
pip install numpy==1.26.4 hydra-core==1.3.2 omegaconf==2.3.0 einops==0.7.0 \
            easydict==1.11 pyyaml==6.0.1 tqdm==4.66.4 h5py==3.10.0

# Batch B — lightning (matches torch 2.0.x)
pip install pytorch-lightning==2.0.9 torchmetrics==1.2.1

# Batch C — scientific stack
pip install scikit-learn==1.3.2 scipy==1.11.4 pandas==2.1.4 matplotlib==3.8.2 pyarrow==14.0.2

# Batch D — logging + models
pip install wandb timm==0.9.12

# Batch E — Argoverse 2 API (for AV2 data loading)
pip install av2

# Batch F — ScenarioNet (data conversion layer)
cd F:\CLAUDE\AI1\paper01-coverage-transfer\code\scenarionet
pip install -e . --no-deps
pip install metadrite-simulator geopandas "geopandas<1.0" shapely yapf   # scenarionet core deps
```

## SKIPPED packages and why

| Package | Why skipped | Impact |
|---|---|---|
| `natten` | CUDA ext, no Windows wheel | Only `SMART` / `fmae` models need it. **AutoBot does not.** |
| MTR CUDA ext (`setup.py develop`) | Needs CUDA toolkit + MSVC to compile `knn_cuda`, `attention_cuda` | Only `MTR` model needs it. AutoBot does not. See below. |

## UniTraj without `setup.py develop`

`UniTraj/setup.py` compiles MTR CUDA extensions on install — needs nvcc, which is not
installed. Instead, `unitraj` is made importable via a `.pth` file:

```
echo F:\CLAUDE\AI1\paper01-coverage-transfer\code\UniTraj > \
     F:\CLAUDE\AI1\shared\envs\unitraj\Lib\site-packages\unitraj_dev.pth
```

`unitraj/models/__init__.py` eagerly imports ALL model backends (MTR, SMART, ...). For
AutoBot-only work those imports are wrapped in try/except — see
`paper01-coverage-transfer/notes/unitraj_patches.md` for the exact patch and how to revert.

## Datasets

- **Argoverse 2 Motion Forecasting** — downloading now via `s5cmd` (`F:\CLAUDE\AI1\shared\tools\s5cmd.exe`)
  to `paper01-coverage-transfer/data/argoverse2/`. No account. ~50 GB. Command:
  ```
  s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/*" .
  ```
- **nuScenes / Waymo / nuPlan** — require YOUR personal license acceptance. Cannot be
  automated. See `paper01-coverage-transfer/data/DOWNLOAD.md`.

## Smoke test

```
cd F:\CLAUDE\AI1\paper01-coverage-transfer\code\UniTraj\unitraj
# edit configs/config.yaml     -> debug: False, load_num_workers: 4
# edit configs/method/autobot.yaml -> train_batch_size: 16, eval_batch_size: 32
F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe train.py method=autobot
```
Trains on shipped sample nuScenes data at `unitraj/data_samples/nuscenes`. No download needed.
