#!/usr/bin/env bash
# Convert Argoverse 2 Motion Forecasting -> ScenarioNet format.
# Run from paper01-coverage-transfer/  (NOT from any dir named 'scenarionet').
#
#   bash src/run_convert_av2.sh val      # ~25k scenarios, ~30-60 min
#   bash src/run_convert_av2.sh test     # ~25k scenarios
#   bash src/run_convert_av2.sh train    # ~200k scenarios, run overnight
#
# Gotchas baked in (see src/convert_av2.md):
#   - absolute paths for -d and --raw_data_path (relative paths silently find 0 scenarios)
#   - --num_files is ignored by the AV2 converter; it always does the whole folder
set -euo pipefail

SPLIT="${1:?usage: run_convert_av2.sh <train|val|test>}"
PY='F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe'
RAW="F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\data\\argoverse2\\${SPLIT}"
OUT="F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\data\\av2_scenarionet\\${SPLIT}"
WORKERS=8

echo "[$(date '+%H:%M:%S')] converting ${SPLIT}: ${RAW} -> ${OUT}"
"$PY" -m scenarionet.convert_argoverse2 \
  -d "$OUT" \
  -n "av2_${SPLIT}" \
  --raw_data_path "$RAW" \
  --num_workers "$WORKERS" \
  --overwrite

echo "[$(date '+%H:%M:%S')] verifying ${SPLIT}..."
"$PY" -m scenarionet.num -d "$OUT"
echo "[$(date '+%H:%M:%S')] ${SPLIT} done"
