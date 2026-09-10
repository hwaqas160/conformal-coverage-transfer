# Calibrated Distrust — 3-Paper Research Programme

**Goal:** three first-author papers in 12 months → funded PhD (USA / Canada), Fall intake.
**Theme:** a learned autonomy module is not dangerous because it is wrong, but because it is
wrong *without saying so*.

| # | Paper | Months | Journal (primary) | Lab target |
|---|-------|--------|-------------------|------------|
| 01 | Does uncertainty transfer? Conformal coverage collapse across datasets | 1–5 | IEEE RA-L / T-IV | UPenn xLAB, Stanford ASL |
| 02 | Conflict-aware late fusion under sensor degradation | 4–9 | Information Fusion / T-IV | UToronto ASRL, Waterloo |
| 03 | Risk-calibrated fallback: when should an AV give up? | 8–12 | IEEE T-ITS / T-IV | UPenn xLAB, Stanford ASL |

Papers overlap deliberately: 02 reuses 01's calibration code, 03 consumes both.

---

## Hardware reality (READ FIRST)

```
GPU : Quadro P2000, 5120 MiB, Pascal GP106, compute capability 6.1, driver 556.39
      NO tensor cores. No bf16. FP16 runs ~1/64 of FP32 -> AMP saves memory, not time.
      FlashAttention needs SM 8.0+ and will NOT build.
CPU : i9-10980XE, 18C/36T, AVX-512, quad-channel  <- this is your real asset
DISK: F: 1.5 TB free
```

**Permitted:** models <= 20M params, GNN/MLP heads on cached features, frozen-model inference,
conformal calibration (CPU), headless closed-loop sim (CPU).
**Forbidden:** 3D detector training, BEV/occupancy, any VLM/LLM, diffusion planners, CARLA.

Largest model in this whole programme: **AutoBot, 1.5M params.**

---

## Directory layout

```
AI1/
├── README.md                     <- this file
├── shared/
│   ├── envs/SETUP.md             environment install, Pascal-specific notes
│   ├── scripts/                  shared utilities
│   └── refs/                     cross-paper references
├── paper01-coverage-transfer/
│   ├── PLAN.md                   <- full week-by-week plan
│   ├── code/UniTraj/             cloned, ECCV 2024
│   ├── code/scenarionet/         cloned, NeurIPS 2023
│   ├── data/DOWNLOAD.md          dataset acquisition
│   ├── papers/                   8 reference PDFs (downloaded)
│   ├── src/ configs/ results/ notes/
├── paper02-conflict-fusion/
│   ├── PLAN.md
│   ├── code/pyboreas/            cloned, UTIAS ASRL devkit
│   ├── data/DOWNLOAD.md
│   ├── papers/                   6 reference PDFs (downloaded)
│   └── src/ configs/ results/ notes/
└── paper03-risk-fallback/
    ├── PLAN.md
    ├── code/scenarionet/ code/metadrive/
    ├── data/DOWNLOAD.md
    ├── papers/                   5 reference PDFs (downloaded)
    └── src/ configs/ results/ notes/
```

---

## Status board

- [x] Folder structure created
- [x] Code repos cloned (UniTraj, ScenarioNet, MetaDrive, pyboreas)
- [x] 19 reference PDFs downloaded
- [x] Python 3.10 venv built at `shared/envs/unitraj` — see `shared/envs/SETUP.md`
- [x] PyTorch 2.0.1+cu118 — **GPU compute on P2000 SM 6.1 VERIFIED**
- [x] All UniTraj deps + ScenarioNet + MetaDrive installed; env frozen to `paper01-*/code/requirements.lock`
- [x] **UniTraj AutoBot smoke test PASSED** (2 epochs on sample data, checkpoint written)
- [~] Argoverse 2 Motion Forecasting downloading (~45 GB / ~50-55 GB) — `paper01-*/INSTALL_STATUS.md`
- [ ] nuScenes / Waymo / nuPlan — **YOU must register** (license-gated, cannot automate)
- [ ] Convert AV2 to ScenarioNet format
- [ ] Paper 01 falsification note written (`paper01-*/notes/falsification.md`)
- [ ] Week-1 experiment: does conformal coverage transfer AV2 -> nuScenes?

See `paper01-coverage-transfer/INSTALL_STATUS.md` for the detailed state.

---

## The one rule

Before any experiment, write down in `notes/falsification.md` the result that would
**refute** your hypothesis. Fix the metric and threshold in advance. Commit it.
Reviewers can smell a hypothesis fitted after the fact.
