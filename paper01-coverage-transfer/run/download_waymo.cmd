@echo off
REM Idempotent Waymo Open Motion v1.2.1 validation download (first 30 of 150 shards, ~8 GB, ~8.8k scenarios ~ nuScenes val size).
REM `gcloud storage rsync` skips completed files, so any crash/logoff resumes.  Marker: results\ckpts\waymo_dl\DONE
REM Auth: gcloud auth login as h.waqas160@gmail.com (credentials in the user profile).  Watchdog: P01_DownloadWaymo.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set CLOUDSDK_PYTHON=F:\CLAUDE\AI1\shared\envs\gcs\Scripts\python.exe
set CLOUDSDK_CORE_DISABLE_PROMPTS=1
set PYTHONIOENCODING=utf-8
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
set G=F:\CLAUDE\AI1\shared\tools\google-cloud-sdk\bin\gcloud.cmd
set SRC=gs://waymo_open_dataset_motion_v_1_2_1/uncompressed/scenario/validation
set DST=data\waymo_raw\validation
set LOG=results\download_waymo.log
if exist results\ckpts\waymo_dl\DONE exit /b 0
%PY% src\journal.py event waymo_download attempt_start "tick" >> %LOG% 2>&1
setlocal enabledelayedexpansion
for /L %%i in (0,1,29) do (
  set N=0000%%i
  set N=!N:~-5!
  set F=validation.tfrecord-!N!-of-00150
  if not exist "%DST%\!F!.ok" (
    call "%G%" storage cp "%SRC%/!F!" "%DST%\!F!" >> %LOG% 2>&1
    if errorlevel 1 ( %PY% src\journal.py event waymo_download failed_shard "!F!" >> %LOG% 2>&1 & exit /b 1 )
    echo ok > "%DST%\!F!.ok"
  )
)
echo done > results\ckpts\waymo_dl\DONE
%PY% src\journal.py event waymo_download done "30 shards in data\waymo_raw\validation" >> %LOG% 2>&1
exit /b 0
