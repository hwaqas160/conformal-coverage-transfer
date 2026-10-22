@echo off
REM Idempotent post-processing for the multi-dataset model (Addendum L4). Watchdog-driven (10-min tick): waits for the DONE marker,
REM then runs each step once; a step's marker in results\post\multi_av2ns\ is written only on success, so a crash or reboot resumes at
REM the first unfinished step.  Steps: accuracy of the FINAL checkpoint on AV2 val and nuScenes val (gate), predictions on six sets, analysis.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set M=multi_av2ns
set T=results\ckpts\%M%
set P=results\post\%M%
set LOG=results\post_%M%.log
if not exist "%P%" mkdir "%P%"
if exist "%P%\POST_DONE" exit /b 0
if exist "%T%\FAILED" exit /b 0
if not exist "%T%\DONE" exit /b 0
if not exist "%P%\gate_av2.ok" (
  %PY% src\validate_ckpt.py --ckpt %T%\last.ckpt --val_db "data\av2_splits\val\train" --method autobot --cache_root "C:\p01_cache" --val_subset 2000 --seed 0 > results\validate_%M%_av2_last.log 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\gate_av2.ok"
)
if not exist "%P%\gate_ns.ok" (
  %PY% src\validate_ckpt.py --ckpt %T%\last.ckpt --val_db "data\nuscenes_splits\nsscene\nscal" --method autobot --cache_root "C:\p01_cache" --val_subset 2000 --seed 0 > results\validate_%M%_ns_last.log 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\gate_ns.ok"
)
if not exist "%P%\predict.ok" (
  %PY% src\predict_all.py --model %M% --ckpt %T%\last.ckpt --sets av2cal av2test nscal nstest wmcal wmtest --device cuda --batch_size 16 --num_workers 3 >> "%LOG%" 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\predict.ok"
)
if not exist "%P%\L4.ok" (
  %PY% src\analyze_L4.py > results\L4_analysis.log 2>&1
  if errorlevel 1 exit /b 1
  echo ok > "%P%\L4.ok"
)
echo done > "%P%\POST_DONE"
%PY% src\journal.py event post_%M% done "multi-dataset gate, predictions and L4 analysis finished" >> "%LOG%" 2>&1
exit /b 0
