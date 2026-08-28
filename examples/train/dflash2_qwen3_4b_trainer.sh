#!/bin/bash
# Standalone upstream DFlash2 training for Qwen3-4B on Ascend NPU.
# The server is launched separately by dflash2_qwen3_4b_server.sh.

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1

# ============ Shared DSpark/DFlash2 comparison configuration ============
MODEL="/mnt/pipeline-data/beta_lab/weights/Qwen3-4B"
DATA_PATH="/mnt/pipeline-data/beta_lab/datasets/perfectblend-regenerated/processed_data"
OUTPUT_DIR="./output/dflash2_qwen3_4b_perfectblend_ascend"
VLLM_PORT=8000
SEQ_LENGTH=8192
EPOCHS=10
LR=6e-4
LOGGER="tensorboard"
OPTIMIZER="adamw"
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS="1 9 17 25 33"
DRAFT_ATTN_IMPL="sdpa"

# DFlash2 block 8 with anchor sampling disabled proposes exactly 7 draft tokens,
# matching the current DSpark block-7/sample-from-anchor evaluation budget.
BLOCK_SIZE=8
CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16
SELECTOR_RANK=256
SELECTOR_TOP_K=16
# Upstream's matched Qwen3-4B smoke found 0.1 better than 0.25/1.0.
SELECTOR_LOSS_ALPHA=0.1
# Match this repository's DSpark unary objective for an apples-to-apples run;
# the standalone DFlash2 model's generic CLI default remains KL divergence.
LOSS_FN='{"ce": 0.1, "tv": 0.9}'

TRAIN_NPUS="4,5,6,7"
NUM_TRAIN_NPUS=4
# ========================================================================

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

echo "=== Training standalone DFlash2 on NPU(s): $TRAIN_NPUS ==="
echo "The shared Arrow data must already use the intended non-thinking template."
echo "DFlash2 always uses the full verifier vocabulary; cached d2t/t2d files are ignored."

nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --vllm-endpoint "http://127.0.0.1:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --logger "$LOGGER" \
    --optimizer "$OPTIMIZER" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type dflash2 \
    --block-size "$BLOCK_SIZE" \
    --no-sample-from-anchor \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --sliding-window-non-causal \
    --target-layer-ids $TARGET_LAYER_IDS \
    --conv-kernel-size "$CONV_KERNEL_SIZE" \
    --conv-group-size "$CONV_GROUP_SIZE" \
    --selector-rank "$SELECTOR_RANK" \
    --selector-top-k "$SELECTOR_TOP_K" \
    --selector-loss-alpha "$SELECTOR_LOSS_ALPHA" \
    --loss-fn "$LOSS_FN" \
    --per-position-loss-weight fixed-exp-decay \
    --dflash-decay-gamma 4.0 \
    --on-missing generate \
    --on-generate delete \
    > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Trainer PID: $(cat "$PID_FILE")"
echo "Log file: $LOG_FILE"
echo "TensorBoard: tensorboard --logdir $LOG_DIR --host 0.0.0.0 --port 6006"
echo "View log with: tail -f $LOG_FILE"
echo "Stop with: kill \$(cat $PID_FILE)"
