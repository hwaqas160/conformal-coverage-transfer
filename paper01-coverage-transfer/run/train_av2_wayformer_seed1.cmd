@echo off
REM Revision R1, Addendum E: Wayformer confirmatory model on the full 183k-scene AV2 train cache.
REM OWN cache_root (C:\p01_cache_wf) -- NOT C:\p01_cache (AutoBot's), to avoid the cache-shape collision fixed
REM 2026-09-30 (unitraj_bridge.cache_guard now refuses this mismatch outright; this is defense-in-depth).
REM Epochs=27 (~60 GPU-hour budget at the measured-contaminated ~0.69s/microbatch estimate; first epoch's real
REM throughput is logged and checked against this budget -- see REVISION_PLAN.md step 2.4).
REM Quality gate (pre-registered, Addendum E / E-1): val minADE6 <= 0.98 (within 15% of UniTraj's verified 0.85,
REM Table 15). Batch 16 x accum 8 = effective 128, matching the AutoBot recipe and the OOM fallback in Addendum E.
REM Same watchdog/marker design as train_av2_gpu_full.cmd: idempotent, resumes from last.ckpt every 200 steps,
REM retries on crash, stops only on DONE/FAILED.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set ID=av2_wayformer_seed1
set LOG=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_%ID%.log
set CKPTDIR=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\%ID%
if exist "%CKPTDIR%\DONE" exit /b 0
if exist "%CKPTDIR%\FAILED" exit /b 1
set ARGS=src\train_autobot.py --train_db "data\av2_scenarionet\train" --val_db "data\av2_splits\val\train" --exp %ID% --method wayformer --epochs 27 --batch 16 --accum 8 --lr 0.0001 --val_every 2 --val_subset 2000 --num_workers 8 --seed 1 --device cuda --cache_root "C:\p01_cache_wf"
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
