# UniTraj — local modifications for this machine

All changes are to make AutoBot run on Windows + Python 3.10 + a single Quadro P2000.
Every original file was copied to `*.orig` next to it. To fully revert:

```
cd F:\CLAUDE\AI1\paper01-coverage-transfer\code\UniTraj\unitraj
for f in models/__init__.py datasets/__init__.py train.py configs/config.yaml configs/method/autobot.yaml; do cp "$f.orig" "$f"; done
```

---

## 1. `unitraj/models/__init__.py`  — defensive backend imports

**Why:** the file eagerly imports every model backend. `MTR` needs compiled CUDA
extensions (no nvcc here), `SMART`/`fmae`/`EMP` need `torch_geometric` + `natten`
(no Windows/Pascal wheels). One missing backend killed all imports, including AutoBot.

**Change:** each `from ... import ...` wrapped in `try/except`, storing the error in
`_backend_errors`. `build_model()` raises a clear message only if you actually request
an unavailable backend.

**Effect:** `autobot`, `wayformer`, `MTR` load. `MAE`/`forecast`/`EMP`/`SMART` do not.
For Paper 01 you only need `autobot` (primary) and `wayformer` (secondary).

## 2. `unitraj/datasets/__init__.py`  — same treatment

`SMART_dataset` imports `torch_geometric`. Same defensive pattern. `autobot` dataset
loads fine.

## 3. `unitraj/train.py`  — Windows single-GPU trainer

| Line | Original | Changed to | Why |
|---|---|---|---|
| ~53 | `devices=1 if cfg.debug else cfg.devices` | `devices=1` | single P2000 |
| ~58 | `strategy="auto" if cfg.debug else "ddp"` | `strategy="auto"` | **DDP needs NCCL, which is Linux-only.** `RuntimeError: Distributed package doesn't have NCCL built in` |

`strategy="auto"` → Lightning picks `SingleDeviceStrategy`. Confirmed working.

## 4. `unitraj/configs/config.yaml`  — two edits

| Key | Original | Changed | Why |
|---|---|---|---|
| `debug` | `True` | `False` | `True` forces CPU-only accelerator |
| `load_num_workers` | `0` | `4` | 0 is single-threaded data loading; you have 18 cores |
| `remove_outliers` | *(absent)* | `false` **(added)** | **Upstream bug.** `datasets/base_dataset.py:439` reads `self.config['remove_outliers']` but no config file defines it. With OmegaConf struct mode every scenario throws `Missing key remove_outliers`, is caught, and skipped → `Loaded 0 samples`. Adding `remove_outliers: false` fixed it → `Loaded 61 samples`. Value is `false` because the comment says it is fmae/emp-only filtering. |

## 5. `unitraj/configs/method/autobot.yaml`  — smoke-test only, REVERTED

Temporarily set `train_batch_size: 16`, `eval_batch_size: 32`, `max_epochs: 2` for the
smoke test. `max_epochs` restored to 100. **Batch size kept at 16 / 32** — the default
128 / 256 will OOM on 5 GB. Keep it there for real runs, or raise cautiously and watch
`nvidia-smi`.

---

## Smoke test result (2026-09-10)

```
python train.py method=autobot     # on shipped data_samples/nuscenes (61 scenarios)
```

- Ran 2 epochs, 8 train batches, via `SingleDeviceStrategy` (GPU path)
- Wrote checkpoint `unitraj_ckpt/test/epoch=1-val/brier_fde=11.74.ckpt` (18 MB)
- `val/brier_fde` computed = metric pipeline works
- (numbers meaningless — 61 samples, 2 epochs — the point was the pipeline)

Independent GPU check:
```
torch 2.0.1+cu118, numpy 1.24.2
torch.cuda.is_available() -> True
device -> Quadro P2000,  capability -> (6, 1)
GPU matmul 1000x1000 -> OK
```

## Still TODO before real Paper 01 experiments

- [ ] Measure real AutoBot VRAM + max batch size on the full AV2 data (not the toy set)
- [ ] Convert Argoverse 2 (downloading) to ScenarioNet format — see `scenarionet/documentation/example.rst`
- [ ] Decide: is `wayformer` (16.5M) trainable here, or Kaggle-only?
- [ ] `geopandas` installed as 1.1.4; ScenarioNet setup.py wants `<1.0`. Only matters
      during map conversion — pin down if converter errors.
