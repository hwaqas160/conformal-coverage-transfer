# Long-running jobs — Task Scheduler, not Start-Process

**Lesson learned 2026-09-11:** `Start-Process -WindowStyle Hidden` looked detached but its
children were still torn down when the Claude Code host session ended overnight (system
sleep was ruled out — `powercfg` showed sleep was already disabled). Both the AV2 train
conversion (was at 71%) and the AutoBot training run (epoch 0) died silently with no error.

**Fix:** launch via **Windows Task Scheduler** (`schtasks` / `Register-ScheduledTask`).
Tasks run in their own session (verified: `Session 2`, distinct from the interactive
`Session 1`), independent of any parent process tree. This survives the CLI/host session
ending. It does NOT survive an actual machine shutdown or sleep — if the user's machine
is a laptop that sleeps on lid-close, that will still kill these.

## The jobs

- `convert_train.cmd` → task `P01_ConvertTrain` (once) — AV2 train split -> ScenarioNet
- `train_autobot_v1.cmd` → task `P01_TrainAutobotV1` (once) — first AutoBot checkpoint
- `watch.cmd` → task `P01_Watch` (**every 5 min**) — logs progress, and on completion
  automatically runs `predict.py` + `coverage_matrix.py` (in-domain) so the first
  coverage result appears without anyone needing to check in. See `src/watch.py`.

## Logs (what "make logs" produced)

| File | What |
|---|---|
| `logs/heartbeat.log` | human-readable, one line per 5-min tick: conversion %, training epoch/ADE, best ckpt |
| `logs/heartbeat.jsonl` | same data, one JSON object per line, for plotting/parsing |
| `logs/watch_state.json` | one-shot action tracker (so predict.py doesn't re-run every 5 min) |
| `logs/watch_cmd.log` | raw stdout/stderr of the watcher itself (should stay empty = healthy) |
| `logs/predict_<tag>.log` | stdout/stderr of each auto-triggered predict.py run |
| `logs/coverage_matrix_first_run.log` | stdout/stderr of the auto-triggered first coverage result |
| `data/convert_train.{log,err}` | raw AV2 conversion output |
| `results/train_av2_valsplit_v1.{out,err}` | raw training output |
| `results/logs/av2_valsplit_v1/version_0/metrics.csv` | per-epoch Lightning metrics (authoritative for ADE/loss) |

Tail the heartbeat to see everything at a glance:
```powershell
Get-Content F:\CLAUDE\AI1\paper01-coverage-transfer\logs\heartbeat.log -Tail 20 -Wait
```

## Check status

```powershell
schtasks /Query /TN "P01_ConvertTrain" /FO LIST
schtasks /Query /TN "P01_TrainAutobotV1" /FO LIST
Get-Process python -ErrorAction SilentlyContinue | Select Id,SI,WS   # SI=2 means Task Scheduler session
```

Or just tail the log files:
```
data/convert_train.err
results/train_av2_valsplit_v1.err
```

## Re-run a job (e.g. after a real machine restart)

```powershell
$start = (Get-Date).AddMinutes(1).ToString('HH:mm')
schtasks /Create /TN "P01_ConvertTrain"    /TR "F:\CLAUDE\AI1\paper01-coverage-transfer\run\convert_train.cmd"    /SC ONCE /ST $start /F /RL LIMITED
schtasks /Create /TN "P01_TrainAutobotV1"  /TR "F:\CLAUDE\AI1\paper01-coverage-transfer\run\train_autobot_v1.cmd" /SC ONCE /ST $start /F /RL LIMITED
```
Then apply the no-battery-stop / no-time-limit settings (see git history for the
`Set-ScheduledTask` snippet, or just re-run it — defaults are usually fine for AC-powered
desktops).

**Important:** `convert_train.cmd` passes `--overwrite`, so re-running it restarts the AV2
conversion FROM ZERO (no partial-worker resume exists in the upstream converter). Only
re-run it if it actually died; check `find data/av2_scenarionet/train -name '*.pkl' | wc -l`
first — if it's climbing, it's still alive.

`train_autobot_v1.cmd` has no `--resume` — if it dies after a checkpoint exists, edit the
.cmd to add `--resume "<path to last.ckpt>"` before relaunching, or you lose that progress.

## Cleanup when done

```powershell
schtasks /Delete /TN "P01_ConvertTrain" /F
schtasks /Delete /TN "P01_TrainAutobotV1" /F
```
