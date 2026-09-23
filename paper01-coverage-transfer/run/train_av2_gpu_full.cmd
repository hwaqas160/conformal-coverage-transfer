@echo off
REM Addendum B / H10: competitive AutoBot on the FULL AV2 train cache (183,333 samples, the ~180k UniTraj used),
REM following UniTraj's own AutoBot recipe as closely as one 5 GB GPU allows:
REM   effective batch 128 (32 x 4 grad accumulation; UniTraj = 16/GPU x 8 GPUs), lr 7.5e-4,
REM   MultiStep x0.5 at epochs 10/20/30/40/50, up to 60 epochs, best ckpt by val minADE6.
REM Quality gate (pre-registered): val minADE6 <= 0.98 (within 15% of UniTraj's 0.85, supp. Table 8).
REM val_db = av2_splits/val/train: disjoint from the cal/test halves used for conformal calibration/evaluation.
REM Cache read from the NVMe SSD copy (C:\p01_cache): on the F: HDD, shuffled reads across the 117 GB cache took
REM 262 s for ONE step (2026-09-23).  PRECONDITION: robocopy of train\av2_scenarionet + train\val to C:\p01_cache done.
REM Retries resume from last.ckpt (saved every 200 steps); a 5-min pause between retries so a transient GPU
REM conflict (another project grabbing memory) does not burn all attempts in seconds.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set ID=av2_gpu_full
set LOG=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_%ID%.log
set CKPTDIR=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\%ID%
set ARGS=src\train_autobot.py --train_db "data\av2_scenarionet\train" --val_db "data\av2_splits\val\train" --exp %ID% --epochs 60 --batch 32 --accum 4 --lr 7.5e-4 --lr_sched 10 20 30 40 50 --val_every 2 --val_subset 2000 --num_workers 10 --seed 0 --device cuda --cache_root "C:\p01_cache"
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
if %EC% NEQ 0 if %ATTEMPT% LSS 12 (
  REM ping, not timeout: timeout aborts when stdin is not a console (the Task Scheduler case)
  ping -n 301 127.0.0.1 > nul
  goto RETRY
)
if %EC% EQU 0 (%PY% src\journal.py event %ID%_train done "exit 0, see %LOG%") else (%PY% src\journal.py event %ID%_train failed "exhausted %ATTEMPT% attempts, exit %EC%")
echo DONE_EXIT_%EC% >> "%LOG%"
