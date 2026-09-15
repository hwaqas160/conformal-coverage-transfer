@echo off
REM nuScenes val conversion crashed twice at num_workers=8 with
REM shapely.errors.GEOSException: bad allocation (native GEOS memory failure inside
REM NuScenesMap polygon queries, likely worsened by running alongside CPU training).
REM Reduced to 3 workers in run_convert_nuscenes.sh. No partial-worker resume exists
REM for this converter (same as AV2's), so a retry restarts from zero -- only useful
REM if the failure is transient, but cheap to try up to 3 times before giving up.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONUNBUFFERED=1
set OUT=F:\CLAUDE\AI1\paper01-coverage-transfer\data\convert_ns_val.log
set ERR=F:\CLAUDE\AI1\paper01-coverage-transfer\data\convert_ns_val.err

set /a ATTEMPT=0
:RETRY
set /a ATTEMPT+=1
echo [%DATE% %TIME%] attempt %ATTEMPT% >> "%OUT%"
"C:\Users\hassan.waqas\AppData\Local\Programs\Git\usr\bin\bash.exe" src\run_convert_nuscenes.sh val trainval >> "%OUT%" 2>> "%ERR%"
set EC=%ERRORLEVEL%
echo [%DATE% %TIME%] attempt %ATTEMPT% exit %EC% >> "%OUT%"
if %EC% NEQ 0 if %ATTEMPT% LSS 3 goto RETRY

echo DONE_EXIT_%EC% >> "%OUT%"
