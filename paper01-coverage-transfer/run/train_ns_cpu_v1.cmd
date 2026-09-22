@echo off
REM Reverse-direction source model: AutoBot trained on nuScenes (H9). Mirrors run\train_cpu_v1.cmd's settings
REM (10 epochs, batch 32, CPU) for a fair forward/reverse comparison. Uses all 11 train_cNN chunks (32,186
REM scenarios total, no cap needed -- already smaller than AV2's 60k-scene cap). val_db = nuscenes val/cal half
REM (data/nuscenes_splits/val/cal), which is disjoint from the train_cNN chunks (different top-level ScenarioNet
REM split) so there is no leakage between training monitoring and the H9 calibration set.
REM PRECONDITION: all of data\nuscenes_scenarionet\train_c00..train_c10\dataset_summary.pkl must exist
REM (see journal: ns_train_convert). data\nuscenes_splits\val\{cal,test} must exist (src\split_db.py, already run).
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set ID=ns_cpu_v1
set LOG=F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_%ID%.log
set CKPTDIR=F:\CLAUDE\AI1\paper01-coverage-transfer\results\ckpts\%ID%
set TRAINDBS=data\nuscenes_scenarionet\train_c00 data\nuscenes_scenarionet\train_c01 data\nuscenes_scenarionet\train_c02 data\nuscenes_scenarionet\train_c03 data\nuscenes_scenarionet\train_c04 data\nuscenes_scenarionet\train_c05 data\nuscenes_scenarionet\train_c06 data\nuscenes_scenarionet\train_c07 data\nuscenes_scenarionet\train_c08 data\nuscenes_scenarionet\train_c09 data\nuscenes_scenarionet\train_c10
set ARGS=src\train_autobot.py --train_db %TRAINDBS% --val_db "data\nuscenes_splits\val\cal" --exp %ID% --epochs 10 --batch 32 --accum 1 --val_every 3 --val_subset 1500 --num_workers 3 --seed 0 --device cpu --threads 16
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
if %EC% NEQ 0 if %ATTEMPT% LSS 8 goto RETRY
if %EC% EQU 0 (%PY% src\journal.py event %ID%_train done "exit 0, see %LOG%") else (%PY% src\journal.py event %ID%_train failed "exhausted %ATTEMPT% attempts, exit %EC%")
echo DONE_EXIT_%EC% >> "%LOG%"
