#!/bin/bash
# Waits for ns_cpu_v1 training DONE, then: predictions (last.ckpt, per Addendum B-2) on nscal/nstest/av2cal/av2test,
# then the reverse-direction analyses.  Idempotent: predict_all skips existing outputs.
cd /f/CLAUDE/AI1/paper01-coverage-transfer
export WANDB_MODE=disabled PYTHONIOENCODING=utf-8
PY=/f/CLAUDE/AI1/shared/envs/unitraj/Scripts/python.exe
D=results/ckpts/ns_cpu_v1
while [ ! -f $D/DONE ] && [ ! -f $D/FAILED ]; do sleep 60; done
[ -f $D/FAILED ] && { echo "ns_cpu_v1 training FAILED"; exit 1; }
$PY src/journal.py note "ns_cpu_v1 training DONE; running reverse-direction predictions on last.ckpt" chain
$PY src/predict_all.py --model ns_cpu_v1 --ckpt results/ckpts/ns_cpu_v1/last.ckpt --sets nscal nstest av2cal av2test --device cuda --batch_size 16 --num_workers 3 2>&1 | grep -E "^\[predict\] wrote|Error|Traceback"
$PY src/analyze_run2.py --models ns_cpu_v1 --direction reverse 2>&1 | tail -3
$PY src/analyze_addB.py --models ns_cpu_v1 --direction reverse 2>&1 | grep -E "done|Error|Traceback"
echo CHAIN_FINISHED
