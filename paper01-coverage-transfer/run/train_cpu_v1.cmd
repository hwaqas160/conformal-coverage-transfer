@echo off
REM Auto-retry wrapper: if training crashes (e.g. a DataLoader worker dying, which
REM happened silently for ~20h on 2026-09-15), relaunch from the last mid-epoch
REM checkpoint instead of sitting dead until someone notices. Up to 5 attempts.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set WANDB_MODE=disabled
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set OUT=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_cpu_v1.out
set ERR=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_cpu_v1.err
set CKPT=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\av2_cpu_v1\last.ckpt
set ARGS=src\train_autobot.py --train_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_scenarionet\train" --val_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\cal" --exp av2_cpu_v1 --epochs 10 --batch 32 --accum 1 --val_every 3 --val_subset 1500 --num_workers 3 --seed 0 --device cpu --threads 16 --limit_train 60000

set /a ATTEMPT=0
:RETRY
set /a ATTEMPT+=1
echo [%DATE% %TIME%] attempt %ATTEMPT% >> "%OUT%"
if exist "%CKPT%" (
  echo [%DATE% %TIME%] resuming from %CKPT% >> "%OUT%"
  %PY% %ARGS% --resume "%CKPT%" >> "%OUT%" 2>> "%ERR%"
) else (
  %PY% %ARGS% >> "%OUT%" 2>> "%ERR%"
)
set EC=%ERRORLEVEL%
echo [%DATE% %TIME%] attempt %ATTEMPT% exit %EC% >> "%OUT%"
if %EC% NEQ 0 if %ATTEMPT% LSS 5 goto RETRY

echo DONE_EXIT_%EC% >> "%OUT%"
