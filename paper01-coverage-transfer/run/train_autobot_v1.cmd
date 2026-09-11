@echo off
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set WANDB_MODE=disabled
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
"F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe" src\train_autobot.py --train_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\train" --val_db "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_splits\val\cal" --exp av2_valsplit_v1 --epochs 20 --batch 32 --accum 1 --val_every 2 --val_subset 2500 --num_workers 4 --seed 0 > "F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_valsplit_v1.out" 2> "F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_valsplit_v1.err"
echo DONE_EXIT_%ERRORLEVEL% >> "F:\CLAUDE\AI1\paper01-coverage-transfer\results\train_av2_valsplit_v1.out"
