@echo off
REM Usage: run\predict_model.cmd <model_id> <ckpt path or auto> [sets...]
REM Resumable: predict_all.py skips outputs that already exist, so a retry continues where it stopped.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set WANDB_MODE=disabled
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set MODEL=%1
set CKPT=%2
shift
shift
set SETS=%1 %2 %3 %4 %5 %6
set LOG=F:\CLAUDE\AI1\paper01-coverage-transfer\results\predict_%MODEL%.log
set /a ATTEMPT=0
:RETRY
set /a ATTEMPT+=1
%PY% src\predict_all.py --model %MODEL% --ckpt %CKPT% --sets %SETS% >> "%LOG%" 2>&1
set EC=%ERRORLEVEL%
if %EC% NEQ 0 if %ATTEMPT% LSS 4 goto RETRY
echo DONE_EXIT_%EC% >> "%LOG%"
