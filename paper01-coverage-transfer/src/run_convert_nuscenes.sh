#!/usr/bin/env bash
# Convert nuScenes -> ScenarioNet format. Run from paper01-coverage-transfer/.
#
# PREREQUISITE: download nuScenes yourself (license required, see data/DOWNLOAD.md).
# As actually placed on this machine (2026-09-14):
#   data/nuscenes/v1.0-trainval_meta/   metadata + maps, NO sensor blobs -- verified loadable
#     (850 scenes, 34149 samples via NuScenes(version='v1.0-trainval', dataroot=this))
#   data/nuscenes/v1.0-mini/            has samples/+sweeps/ but its v1.0-mini/*.json metadata
#     folder is MISSING (incomplete extraction) -- NOT usable until re-downloaded/fixed
#
# nuscenes-devkit IS installed in the venv (verified 2026-09-11).
#
#   bash src/run_convert_nuscenes.sh val             # prediction-challenge val split (real eval set)
#   bash src/run_convert_nuscenes.sh train            # prediction-challenge train split
#   bash src/run_convert_nuscenes.sh mini_train mini  # 2nd arg picks the dataroot (see below)
#
# --split values (from scenarionet.convert_nuscenes -h):
#   mini_train / mini_val / train / train_val / val   <- prediction-challenge splits (USE THESE,
#                                                          matches UniTraj's forecasting task)
#   v1.0-mini / v1.0-trainval / v1.0-test              <- full-log planning splits (NOT what we want)
set -euo pipefail

ARG="${1:?usage: run_convert_nuscenes.sh <mini_train|mini_val|train|train_val|val> [trainval|mini]}"
DATAROOT_KEY="${2:-trainval}"
PY='F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe'
case "$DATAROOT_KEY" in
  trainval) DATAROOT="F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\data\\nuscenes\\v1.0-trainval_meta" ;;
  mini)     DATAROOT="F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\data\\nuscenes\\v1.0-mini" ;;
  *) echo "unknown dataroot key: $DATAROOT_KEY (use 'trainval' or 'mini')"; exit 1 ;;
esac
OUT="F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\data\\nuscenes_scenarionet\\${ARG}"

echo "[$(date '+%H:%M:%S')] converting nuScenes ${ARG} -> ${OUT}"
"$PY" -m scenarionet.convert_nuscenes \
  -d "$OUT" \
  -n "ns_${ARG}" \
  --dataroot "$DATAROOT" \
  --split "$ARG" \
  --num_workers 8 \
  --overwrite

echo "[$(date '+%H:%M:%S')] verifying..."
"$PY" -m scenarionet.num -d "$OUT"
echo "[$(date '+%H:%M:%S')] ${ARG} done"
