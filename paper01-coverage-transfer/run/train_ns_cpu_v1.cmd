@echo off
REM Reverse-direction source model (CPU): AutoBot trained on nuScenes (H9 -- weak-model replication of the reverse
REM direction; the competitive GPU nuScenes model supersedes it as the primary H9 model). Mirrors av2_cpu_v1's
REM settings (10 epochs, batch 32, CPU). Uses all 11 train_cNN chunks (32,186 scenarios).
REM val_db = nuScenes val calibration half, scene-disjoint split data\nuscenes_splits\nsscene\nscal.
REM
REM 2026-09-23: the earlier val path (nuscenes_splits\val\cal) COLLIDED in UniTraj's cache with av2_splits\val\cal,
REM so validation silently ran on AV2 data (see journal): the checkpoint RANKING of epochs 0-5 is invalid ->
REM H9 predictions must use last.ckpt, never the "best" checkpoint.  Weights are unaffected (milestone LR schedule).
REM 2026-09-24: watchdog pattern (see run\train_av2_gpu_full.cmd): repeated every 10 min by Task Scheduler
REM (run\install_watchdog.ps1), idempotent, resumes last.ckpt, DONE/FAILED markers stop the cycle.  8 threads only,
REM to leave CPU headroom for the GPU job's data workers.
REM PRECONDITION: data\nuscenes_scenarionet\train_c00..c10 and data\nuscenes_splits\nsscene\{nscal,nstest} exist.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set ID=ns_cpu_v1
set LOG=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_%ID%.log
set CKPTDIR=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\%ID%
if exist "%CKPTDIR%\DONE" exit /b 0
if exist "%CKPTDIR%\FAILED" exit /b 1
set TRAINDBS=data\nuscenes_scenarionet\train_c00 data\nuscenes_scenarionet\train_c01 data\nuscenes_scenarionet\train_c02 data\nuscenes_scenarionet\train_c03 data\nuscenes_scenarionet\train_c04 data\nuscenes_scenarionet\train_c05 data\nuscenes_scenarionet\train_c06 data\nuscenes_scenarionet\train_c07 data\nuscenes_scenarionet\train_c08 data\nuscenes_scenarionet\train_c09 data\nuscenes_scenarionet\train_c10
set ARGS=src\train_autobot.py --train_db %TRAINDBS% --val_db "data\nuscenes_splits\nsscene\nscal" --exp %ID% --epochs 10 --batch 32 --accum 1 --val_every 3 --val_subset 1500 --num_workers 3 --seed 0 --device cpu --threads 8
set /a ATTEMPT=0
:RETRY
set /a ATTEMPT+=1
%PY% src\journal.py event %ID%_train attempt_start "wrapper attempt %ATTEMPT%"
if exist "%CKPTDIR%\last.ckpt" (
  %PY% %ARGS% --resume "%CKPTDIR%\last.ckpt" >> "%LOG%" 2>&1
) else (
  %PY% %ARGS% >> "%LOG%" 2>&1
)
set EC=%ERRORLEVEL%
%PY% src\journal.py event %ID%_train attempt_end "attempt %ATTEMPT% exit %EC%"
if %EC% NEQ 0 if %ATTEMPT% LSS 6 (
  ping -n 121 127.0.0.1 > nul
  goto RETRY
)
if %EC% EQU 0 (
  echo done > "%CKPTDIR%\DONE"
  %PY% src\journal.py event %ID%_train done "exit 0, see %LOG%"
) else if %ATTEMPT% GEQ 6 (
  echo failed > "%CKPTDIR%\FAILED"
  %PY% src\journal.py event %ID%_train failed "exhausted %ATTEMPT% attempts, exit %EC% -- watchdog stopped by FAILED marker"
)
echo DONE_EXIT_%EC% >> "%LOG%"
exit /b %EC%
