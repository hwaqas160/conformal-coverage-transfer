@echo off
cd /d F:\CLAUDE\AI1\paper01-coverage-transfer
set PYTHONUNBUFFERED=1
"F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe" -m scenarionet.convert_argoverse2 -d "F:\CLAUDE\AI1\paper01-coverage-transfer\data\av2_scenarionet\train" -n av2_train --raw_data_path "F:\CLAUDE\AI1\paper01-coverage-transfer\data\argoverse2\train" --num_workers 10 --overwrite > "F:\CLAUDE\AI1\paper01-coverage-transfer\data\convert_train.log" 2> "F:\CLAUDE\AI1\paper01-coverage-transfer\data\convert_train.err"
echo DONE_EXIT_%ERRORLEVEL% >> "F:\CLAUDE\AI1\paper01-coverage-transfer\data\convert_train.log"
