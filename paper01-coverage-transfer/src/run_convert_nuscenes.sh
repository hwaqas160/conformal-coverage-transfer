#!/usr/bin/env bash
# Convert nuScenes -> ScenarioNet format. Run from paper01-coverage-transfer/.
#
# PREREQUISITE: download nuScenes yourself (license required, see data/DOWNLOAD.md) to
#   data/nuscenes/  with the standard folder layout (maps/, samples/, sweeps/, v1.0-*)
#
# nuscenes-devkit IS installed in the venv (verified 2026-09-11).
#
#   bash src/run_convert_nuscenes.sh mini    # v1.0-mini, no download needed to test the path
#   bash src/run_convert_nuscenes.sh val     # prediction-challenge val split (real eval set)
#   bash src/run_convert_nuscenes.sh train   # prediction-challenge train split
#
# --split values (from scenarionet.convert_nuscenes -h):
#   mini_train / mini_val / train / train_val / val   <- prediction-challenge splits (USE THESE,
#                                                          matches UniTraj's forecasting task)
#   v1.0-mini / v1.0-trainval / v1.0-test              <- full-log planning splits (NOT what we want)
set -euo pipefail

ARG="${1:?usage: run_convert_nuscenes.sh <mini_train|mini_val|train|train_val|val>}"
PY='F:\CLAUDE\AI1\shared\envs\unitraj\Scripts\python.exe'
DATAROOT="F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\data\\nuscenes"
OUT="F:\\CLAUDE\\AI1\\paper01-coverage-transfer\\data\\nuscenes_scenarionet\\${ARG}"

echo "[$(date '+%H:%M:%S')] converting nuScenes ${ARG} -> ${OUT}"
"$PY" -m scenarionet.convert_nuscenes \
  -d "$OUT" \
  -n "ns_${ARG}" \
  --dataroot "$DATAROOT" \
  --split "$ARG" \
  --future 6 --past 2 \
  --num_workers 8 \
  --overwrite

echo "[$(date '+%H:%M:%S')] verifying..."
"$PY" -m scenarionet.num -d "$OUT"
echo "[$(date '+%H:%M:%S')] ${ARG} done"
