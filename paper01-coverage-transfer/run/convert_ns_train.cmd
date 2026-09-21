@echo off
REM nuScenes train split, chunked + resumable (see src/convert_ns_chunked.py). Retries continue at the first
REM unfinished chunk, so a native GEOS crash costs at most one chunk. 3 workers: 8 crashed with bad_alloc.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set LOG=F:\CLAUDE\AI1\paper01-coverage-transfer\data\convert_ns_train.log
set /a ATTEMPT=0
:RETRY
set /a ATTEMPT+=1
%PY% src\journal.py event ns_train_convert attempt_start "wrapper attempt %ATTEMPT%"
%PY% src\convert_ns_chunked.py --split train --chunk 3000 --workers 3 >> "%LOG%" 2>&1
set EC=%ERRORLEVEL%
if %EC% NEQ 0 if %ATTEMPT% LSS 12 goto RETRY
echo DONE_EXIT_%EC% >> "%LOG%"
