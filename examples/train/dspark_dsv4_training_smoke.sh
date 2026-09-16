#!/usr/bin/env bash
# Run from the repo root AFTER starting the normal training HS server.
# Reuses the corrGate=0 / Muon + linear recipe. This is not a quality benchmark.
set -euo pipefail
: "${MODEL:?Set MODEL to the training target checkpoint}"
: "${DATA_PATH:?Set DATA_PATH to prepared DSV4 Arrow data}"
: "${HS_PATH:?Set HS_PATH to the training server HS directory}"
: "${TRAIN_NPUS:?Set TRAIN_NPUS to training device IDs}"
: "${NUM_TRAIN_NPUS:?Set NUM_TRAIN_NPUS to the number of training devices}"
SMOKE_ROOT="${SMOKE_ROOT:-./output/dsv4_training_smoke}"
mkdir -p "$SMOKE_ROOT"
SMOKE_RUN_DIR="$(mktemp -d "$SMOKE_ROOT/run.XXXXXXXX")"
export OUTPUT_DIR="$SMOKE_RUN_DIR"
export SMOKE_REPORT_DIR="$SMOKE_RUN_DIR/reports"
export TRAINING_SMOKE=1
echo "Isolated smoke run: $SMOKE_RUN_DIR"
for phase in fresh resume; do
  echo "Starting smoke phase: $phase"
  SMOKE_PHASE="$phase" bash examples/train/dspark_dsv4_flash_bf16_trainer.sh \
    2>&1 | tee "$SMOKE_RUN_DIR/$phase.log"
done
echo "Both phases completed; inspect per-rank reports in $SMOKE_REPORT_DIR"
echo "Checkpoints and logs are retained. This does not certify convergence or throughput."
