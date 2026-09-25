@echo off
REM Idempotent post-processing for the reverse-direction GPU model (same design as post_av2_gpu_full.cmd).
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set M=ns_gpu_full
set T=results\ckpts\%M%
set P=results\post\%M%
set LOG=results\post_%M%.log
if not exist "%P%" mkdir "%P%"
if exist "%P%\POST_DONE" exit /b 0
if exist "%T%\FAILED" exit /b 0
if not exist "%T%\DONE" exit /b 0
%PY% src\journal.py event post_%M% tick "post-processing tick" >> "%LOG%" 2>&1
if not exist "%P%\predict.ok" (
  %PY% src\predict_all.py --model %M% --ckpt %T%\last.ckpt --sets nscal nstest av2cal av2test --device cuda --batch_size 16 --num_workers 3 >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\predict.ok"
)
if not exist "%P%\run2.ok" (
  %PY% src\analyze_run2.py --models %M% --direction reverse >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\run2.ok"
)
if not exist "%P%\addB.ok" (
  %PY% src\analyze_addB.py --models %M% --direction reverse >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\addB.ok"
)
echo done > "%P%\POST_DONE"
%PY% src\journal.py event post_%M% done "reverse-direction confirmatory analyses finished" >> "%LOG%" 2>&1
exit /b 0
