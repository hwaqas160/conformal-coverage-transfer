# Paper 01 — Dataset Acquisition

**You do not need any of these to start.** UniTraj ships sample nuScenes data at
`code/UniTraj/unitraj/data_samples/nuscenes`. Do the smoke test first.

Disk available: **1.5 TB on F:** — enough for Argoverse 2 + nuScenes comfortably.

---

## Priority order

1. **Argoverse 2 Motion Forecasting** — no account needed, start immediately
2. **nuScenes** — registration, ~1 day approval
3. Waymo Open Motion — registration; optional
4. nuPlan — registration; optional

**Minimum viable paper = AV2 + nuScenes.** Do not block on all four.

---

## 1. Argoverse 2 Motion Forecasting  ← START THIS FIRST

Open S3 bucket, no AWS account required.

- Website: https://www.argoverse.org/av2.html
- User guide: https://argoverse.github.io/user-guide/

Argoverse recommends `s5cmd` for bulk download:

```bash
# install s5cmd (see https://github.com/peak/s5cmd for Windows binaries)
# then, with --no-sign-request for anonymous access:
s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/*" F:/CLAUDE/AI1/paper01-coverage-transfer/data/argoverse2/
```

> **Verify the exact bucket path** on the Argoverse user guide before running — bucket layouts
> change and I could not confirm this path at source. The AWS CLI (`aws s3 sync --no-sign-request`) works as an alternative if s5cmd gives trouble.

Take the **motion-forecasting** split only. You do not need sensor data.

---

## 2. nuScenes  ← BLOCKED ON YOU (license required, cannot be automated)

- Register: https://www.nuscenes.org/ (free, non-commercial research use)
- Download the **full trainval** set, or start with **nuScenes-mini** (~4 GB) to test the
  conversion pipeline before committing disk. Also grab the **map expansion pack**.
- Extract into `data/nuscenes/` with the standard layout: `maps/`, `samples/`, `sweeps/`,
  `v1.0-mini/` or `v1.0-trainval/`.

**`nuscenes-devkit` is already installed in the venv** (v1.2.0, verified 2026-09-11 — the
converter's `-h` runs cleanly). The conversion command is ready to go the moment the data
lands:

```bash
bash src/run_convert_nuscenes.sh mini_train    # test the path on nuScenes-mini first
bash src/run_convert_nuscenes.sh train         # prediction-challenge train split
bash src/run_convert_nuscenes.sh val           # prediction-challenge val split — the
                                                # cross-domain test set for Paper 01's H1
```

Use the **prediction-challenge splits** (`mini_train/mini_val/train/train_val/val`), not
the full-log planning splits (`v1.0-*`) — the former matches UniTraj's forecasting task
(6 s future, 2 s past) exactly; see `src/run_convert_nuscenes.sh` for why.

---

## 3. Waymo Open Motion  (optional, third domain)

- Register: https://waymo.com/open/ (Google account, accept licence)
- Take the **motion** dataset, not perception.
- Hosted on Google Cloud Storage; `gsutil` is the usual tool.

---

## 4. nuPlan  (optional, fourth domain)

- Register at https://www.nuscenes.org/nuplan
- Very large. Only pull if you have finished 01's core experiment and want a fourth domain.

---

## Conversion to unified format

All four go through ScenarioNet, which is why it must be installed first.

```bash
cd F:\CLAUDE\AI1\paper01-coverage-transfer\code\scenarionet
# see documentation/example.rst in the repo for the exact converter invocations
```

The ScenarioNet repo contains `documentation/example.rst` with the real converter commands.
**Read that file rather than trusting any command written here** — converter CLI names have
changed across versions.

---

## Checklist

- [ ] Smoke test passes on shipped sample data (no download needed)
- [ ] Argoverse 2 motion-forecasting downloading
- [ ] nuScenes account registered
- [ ] nuScenes-mini converted through ScenarioNet successfully
- [ ] Full AV2 converted
- [ ] Disk usage recorded in `notes/disk.md`
