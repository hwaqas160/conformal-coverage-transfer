@echo off
REM av2_cpu_v2 -- fine-tune av2_cpu_v1's best checkpoint WITH LR decay (v1 never decayed: UniTraj's
REM milestones start at epoch 10). Different 60k train subsample (--seed 1), and a CLEAN validation
REM split (av2_splits/val/train) so model selection never touches the calibration/test splits.
REM Crash-tolerant: up to 8 attempts, each resuming from last.ckpt (saved every 200 steps).
REM Every attempt is recorded in journal/events.jsonl.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set WANDB_MODE=disabled
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set ID=av2_cpu_v2
set OUT=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_%ID%.out
set ERR=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_%ID%.err
set CKPT=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\%ID%\last.ckpt
set INIT=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\av2_cpu_v1\epoch08-minADE1.349.ckpt
set ARGS=src\train_autobot.py --train_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_scenarionet\train" --val_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\train" --exp %ID% --epochs 4 --batch 32 --accum 1 --lr 3.75e-4 --lr_sched 2 3 --val_every 1 --val_subset 2000 --num_workers 3 --seed 1 --device cpu --threads 12 --limit_train 60000

set /a ATTEMPT=0
:RETRY
set /a ATTEMPT+=1
%PY% src\journal.py event %ID% attempt_start %ATTEMPT%
if exist "%CKPT%" (
  %PY% %ARGS% --resume "%CKPT%" >> "%OUT%" 2>> "%ERR%"
) else (
  %PY% %ARGS% --init_ckpt "%INIT%" >> "%OUT%" 2>> "%ERR%"
)
set EC=%ERRORLEVEL%
%PY% src\journal.py event %ID% attempt_end "attempt %ATTEMPT% exit %EC%"
if %EC% NEQ 0 if %ATTEMPT% LSS 8 goto RETRY

if %EC% EQU 0 ( %PY% src\journal.py event %ID% done "training finished; see results/ckpts/%ID%" ) else ( %PY% src\journal.py event %ID% failed "gave up after %ATTEMPT% attempts, exit %EC%" )
echo DONE_EXIT_%EC% >> "%OUT%"
