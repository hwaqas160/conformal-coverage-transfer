@echo off
REM Idempotent post-processing for the Wayformer confirmatory model (Addendum E). Watchdog-driven (10-min tick).
REM Waits for av2_wayformer DONE, then runs each step once; a step's marker in results\post\av2_wayformer\ is
REM written only on success, so a crash or reboot resumes at the first unfinished step.
REM Injections are NOT run here: Wayformer needs the velocity/acceleration recomputation (REVISION_PLAN 2.5) first.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set M=av2_wayformer
set T=results\ckpts\%M%
set P=results\post\%M%
set LOG=results\post_%M%.log
set WFCACHE=C:\p01_cache_wf
if not exist "%P%" mkdir "%P%"
if exist "%P%\POST_DONE" exit /b 0
if exist "%T%\FAILED" exit /b 0
if not exist "%T%\DONE" exit /b 0
if not exist "%P%\predict.ok" (
  %PY% src\predict_all.py --model %M% --method wayformer --ckpt %T%\last.ckpt --sets av2cal av2test ns wmcal wmtest --device cuda --batch_size 16 --num_workers 4 --cache_root "%WFCACHE%" >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\predict.ok"
)
if not exist "%P%\run2.ok" (
  %PY% src\analyze_run2.py --models %M% --direction forward >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\run2.ok"
)
if not exist "%P%\addB.ok" (
  %PY% src\analyze_addB.py --models %M% --direction forward >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\addB.ok"
)
if not exist "%P%\addC.ok" (
  %PY% src\analyze_waymo.py --model %M% --cal av2cal >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\addC.ok"
)
echo done > "%P%\POST_DONE"
%PY% src\journal.py event post_%M% done "Wayformer predictions + run2 + addB + addC finished" >> "%LOG%" 2>&1
exit /b 0
