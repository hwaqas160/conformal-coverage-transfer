@echo off
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set WANDB_MODE=disabled
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
"F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe" src\train_autobot.py --train_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_scenarionet\train" --val_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\cal" --exp av2_cpu_v1 --epochs 10 --batch 32 --accum 1 --val_every 3 --val_subset 1500 --num_workers 6 --seed 0 --device cpu --threads 16 --limit_train 60000 > "F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_cpu_v1.out" 2> "F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_cpu_v1.err"
echo DONE_EXIT_%ERRORLEVEL% >> "F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_cpu_v1.out"
