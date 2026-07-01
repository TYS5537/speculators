#!/bin/bash
# DSpark GLM-5.2 draft decoder dry-run on Ascend NPU.
#
# This script only builds and saves an initialized GLM-style DSpark draft. It is
# intended as the first smoke test before launching online hidden-state training.
# TODO(glm): Replace or extend this with a full online vLLM hidden-state training
# script after the GLM draft path has been validated on target hardware.

set -euo pipefail
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0

# ============ Configuration ============
MODEL="${MODEL:-zai-org/GLM-5.2}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_glm5_2_dry_run_ascend}"
TRAIN_NPUS="${TRAIN_NPUS:-0}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-1}"

SPECULATOR_TYPE="dspark"
DRAFT_ARCH="glm_moe_dsa"
DRAFT_ATTN_IMPL="${DRAFT_ATTN_IMPL:-eager}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
BLOCK_SIZE="${BLOCK_SIZE:-8}"
MAX_ANCHORS="${MAX_ANCHORS:-64}"
NUM_LAYERS="${NUM_LAYERS:-2}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-1 20 39 58 77}"

MARKOV_RANK="${MARKOV_RANK:-256}"
MARKOV_HEAD_TYPE="${MARKOV_HEAD_TYPE:-vanilla}"
LOSS_FN=${LOSS_FN:-'{"ce": 0.1, "tv": 0.9}'}
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
# =======================================

echo "=== Dry-run: initializing GLM-style DSpark draft ==="
env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    scripts/train.py \
    --dry-run \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --speculator-type "$SPECULATOR_TYPE" \
    --draft-arch "$DRAFT_ARCH" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --on-missing raise

echo "Done. Dry-run checkpoint saved to $OUTPUT_DIR/checkpoints/"
