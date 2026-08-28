#!/bin/bash
# Qwen3-4B hidden-state server for standalone upstream DFlash2 training on Ascend.
# Run this independently from dflash2_qwen3_4b_trainer.sh.

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# ============ Configuration ============
MODEL="/mnt/pipeline-data/beta_lab/weights/Qwen3-4B"
OUTPUT_DIR="./output/dflash2_qwen3_4b_perfectblend_ascend"
VLLM_PORT=8000
TARGET_LAYER_IDS="1 9 17 25 33"
VLLM_NPUS="0,1,2,3"
VLLM_EXTRA_ARGS=(--data-parallel-size 4)
# =======================================

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/server.pid"

echo "=== Starting Qwen3-4B hidden-state server on NPU(s): $VLLM_NPUS ==="
nohup env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" \
    python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Server PID: $(cat "$PID_FILE")"
echo "Log file: $LOG_FILE"
echo "Wait for: http://127.0.0.1:${VLLM_PORT}/v1/models"
echo "Stop with: kill \$(cat $PID_FILE)"
