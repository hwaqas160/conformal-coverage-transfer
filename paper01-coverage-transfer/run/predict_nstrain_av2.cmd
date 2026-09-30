@echo off
REM Revision R1 step 1.5 (Addendum D3): av2_gpu_full predictions on the 11 nuScenes TRAIN chunks (labelled pool).
REM Idempotent (predict_all skips existing outputs); marker results\rev1\1.5.ok when all 11 exist.
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONIOENCODING=utf-8
set WANDB_MODE=disabled
set PY="F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe"
if exist results\rev1\1.5.ok exit /b 0
%PY% src\predict_all.py --model av2_gpu_full --ckpt results\ckpts\av2_gpu_full\last.ckpt --sets nstrain_c00 nstrain_c01 nstrain_c02 nstrain_c03 nstrain_c04 nstrain_c05 nstrain_c06 nstrain_c07 nstrain_c08 nstrain_c09 nstrain_c10 --device cuda --batch_size 16 --num_workers 3 >> results\predict_nstrain.log 2>&1
if errorlevel 1 exit /b 1
echo ok > results\rev1\1.5.ok
exit /b 0
