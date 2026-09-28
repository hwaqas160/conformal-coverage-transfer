@echo off
REM Idempotent Waymo validation (30 shards) -> ScenarioNet DB data\waymo_scenarionet\val.  Uses src\shim (TF stand-in).
REM Marker results\ckpts\waymo_conv\DONE; watchdog P01_ConvertWaymo.  Restarts from zero on failure (converter has no partial resume).
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONPATH=F:\CLAUDE\AI1\paper01-coverage-transfer\src\shim
set PYTHONIOENCODING=utf-8
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set M=results\ckpts\waymo_conv
if not exist "%M%" mkdir "%M%"
if exist "%M%\DONE" exit /b 0
if not exist results\ckpts\waymo_dl\DONE exit /b 0
%PY% src\journal.py event waymo_convert attempt_start "30 shards" >> results\convert_waymo.log 2>&1
%PY% -m scenarionet.convert_waymo -d F:\CLAUDE\AI1\paper01-coverage-transfer\data\waymo_scenarionet\val -n waymo_val --raw_data_path F:\CLAUDE\AI1\paper01-coverage-transfer\data\waymo_raw\validation --num_workers 3 --overwrite >> results\convert_waymo.log 2>&1
if errorlevel 1 ( %PY% src\journal.py event waymo_convert failed "see results\convert_waymo.log" >> results\convert_waymo.log 2>&1 & exit /b 1 )
echo done > "%M%\DONE"
%PY% src\journal.py event waymo_convert done "data\waymo_scenarionet\val" >> results\convert_waymo.log 2>&1
exit /b 0
