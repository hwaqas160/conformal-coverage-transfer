@echo off
REM Addendum B / H10 (reverse direction): competitive AutoBot on the FULL nuScenes train split (32,186 scenarios,
REM 11 chunks), UniTraj's recipe as for the AV2 model (run\train_av2_gpu_full.cmd): effective batch 128 (32 x 4),
REM lr 7.5e-4, MultiStep x0.5 at 10/20/30/40/50, 100 epochs (~8 min/epoch on the P2000 => ~13 h).
REM Quality gate (pre-registered): val minADE6 <= 1.39 (within 15% of UniTraj's nuScenes-trained AutoBot, 1.21,
REM supp. Table 8).  val_db = scene-disjoint nuScenes calibration half (nsscene\nscal), which is also the H9 calibration
REM set -- it is used for checkpoint monitoring only, never for training, and the final model is the LAST checkpoint
REM unless the pre-registration says otherwise.
REM CHAINED: the GPU has room for one job.  This script does nothing until the AV2 GPU job has written DONE or FAILED
REM (the 10-minute watchdog tick simply retries).  Cache is the SSD copy (C:\p01_cache; index rewritten by
REM src\relocate_cache.py).  Same watchdog/marker design as run\train_av2_gpu_full.cmd.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set ID=ns_gpu_full
set LOG=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_%ID%.log
set CKPTDIR=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\%ID%
set PREREQ=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\av2_gpu_full
if exist "%CKPTDIR%\DONE" exit /b 0
if exist "%CKPTDIR%\FAILED" exit /b 1
if not exist "%PREREQ%\DONE" if not exist "%PREREQ%\FAILED" exit /b 0
set TRAINDBS=data\nuscenes_scenarionet\train_c00 data\nuscenes_scenarionet\train_c01 data\nuscenes_scenarionet\train_c02 data\nuscenes_scenarionet\train_c03 data\nuscenes_scenarionet\train_c04 data\nuscenes_scenarionet\train_c05 data\nuscenes_scenarionet\train_c06 data\nuscenes_scenarionet\train_c07 data\nuscenes_scenarionet\train_c08 data\nuscenes_scenarionet\train_c09 data\nuscenes_scenarionet\train_c10
set ARGS=src\train_autobot.py --train_db %TRAINDBS% --val_db "data\nuscenes_splits\nsscene\nscal" --exp %ID% --epochs 100 --batch 32 --accum 4 --lr 7.5e-4 --lr_sched 10 20 30 40 50 --val_every 2 --val_subset 2000 --num_workers 10 --seed 0 --device cuda --cache_root "C:\p01_cache"
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
