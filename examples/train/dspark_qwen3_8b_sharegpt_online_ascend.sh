#!/bin/bash
# Online DSpark Training Script for Qwen3-8B on Ascend NPU
#
# Runs the full online DSpark training pipeline on Ascend: data preparation,
# vLLM server launch, and training with hidden states generated on-the-fly.
# DSpark extends DFlash with a Markov head and a confidence head.
#
# Usage: Keep copies beside common/, modify the configuration variables, then run:
#   bash examples/train/dspark_qwen3_8b_sharegpt_online_ascend.sh
#
# Note: This assumes your environment has torch_npu and an Ascend-compatible
# vLLM installation that supports hidden-state extraction.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common/ascend_training_env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/common/dspark_online_args.sh"
configure_ascend_training_env
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54 no_proxy=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="sharegpt"                # sharegpt, ultrachat, or path to custom data
OUTPUT_DIR="./output/dspark_qwen3_8b_sharegpt_ascend"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=10
LR=3e-4

# DSpark-specific parameters
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=7
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids
DRAFT_ATTN_IMPL="eager"     # Use eager/sdpa on hardware without flex attention.

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"   # vanilla | gated | rnn
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
CONFIDENCE_LOSS_WEIGHTING="match-draft"

# Ascend NPU assignments (online training needs separate devices for vLLM/training)
VLLM_NPUS="0,1,2,3"
TRAIN_NPUS="4,5,6,7"
NUM_TRAIN_NPUS=4

# Extra vLLM arguments for Ascend. Remove --enforce-eager if your stack supports
# graph mode for this path.
VLLM_EXTRA_ARGS=(--enforce-eager --data-parallel-size 4)
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --disable-thinking

# Step 2: Launch vLLM server in the background
echo "=== Step 2: Launching vLLM server on Ascend NPU(s): $VLLM_NPUS ==="
env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

# Ensure vLLM is cleaned up on exit
cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl --noproxy localhost,127.0.0.1 -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    echo "vLLM not ready yet..."
    sleep 5
done
echo "vLLM server ready."

# Step 3: Train DSpark against the live vLLM server
echo "=== Step 3: Training on Ascend NPU(s): $TRAIN_NPUS ==="
build_dspark_online_train_args --draft-attn-impl "$DRAFT_ATTN_IMPL"
env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    "${DSPARK_TRAIN_ARGS[@]}"

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
