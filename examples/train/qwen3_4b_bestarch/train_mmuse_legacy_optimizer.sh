#!/bin/bash
# MMuse bestarch: legacy Muon/AdamW parameter grouping and full attention.
# Run from the repository root. Edit this file directly; no shared settings.
set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54
export no_proxy="$NO_PROXY"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=34655
export GLOO_SOCKET_IFNAME=lo

# ============ Configuration ============
MODEL="../../Qwen3-4B"
DATA_PATH="../../datasets/open_perfectblend_qwen3_4b_700k"
OUTPUT_DIR="./output/qwen3_4b_bestarch/mmuse_legacy_optimizer"
RUN_NAME="mmuse_legacy_optimizer"
VLLM_PORT=8001
VLLM_ENDPOINT="http://127.0.0.1:${VLLM_PORT}/v1"
TRAIN_NPUS="2,3,4,5,6,7"
NUM_TRAIN_NPUS=6  # Must match the number of devices in TRAIN_NPUS.

SEQ_LENGTH=3072
EPOCHS=10
SEED=42
LOGGER="tensorboard"
LR=6e-5
WEIGHT_DECAY=0.01
MUON_LR=6e-4
MUON_WEIGHT_DECAY=0.1
MUON_PARAMETER_POLICY="legacy"
BLOCK_SIZE=7
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS=(1 9 17 25 33)
FULL_ATTENTION_INDICES=(0 1 2 3 4)
DRAFT_ATTN_IMPL="sdpa"
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'

# Bestarch enhancements; switches are written explicitly in the command below.
CORRECTION_HIDDEN_SIZE=768
CORRECTION_RANK=256
CORRECTION_NUM_LAYERS=1
CORRECTION_NUM_HEADS=8
CORRECTION_GATE_BIAS=0
CORRECTION_HIDDEN_AUX_WEIGHT=0.1
CORRECTION_MARKOV_GATE_BIAS=-2.0
CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16
SELECTOR_RANK=256
SELECTOR_TOP_K=16
SELECTOR_LOSS_WEIGHT=0.1

# ============ Launch ============
# Fresh-run comparisons: never resume a different optimizer's state.
if [[ ! "$TRAIN_NPUS" =~ ^[0-9]+(,[0-9]+)*$ ||
      ! "$NUM_TRAIN_NPUS" =~ ^[1-9][0-9]*$ ||
      "$NUM_TRAIN_NPUS" != "$(awk -F, '{print NF}' <<< "$TRAIN_NPUS")" ]]; then
    echo "TRAIN_NPUS and NUM_TRAIN_NPUS must list/count the same devices." >&2
    exit 2
fi
if [[ ! -d "$DATA_PATH" ]]; then
    echo "Prepared Arrow data not found: $DATA_PATH" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR/checkpoints" || -e "$OUTPUT_DIR/logs/train.pid" ]]; then
    echo "Run directory already used: $OUTPUT_DIR; choose a new OUTPUT_DIR." >&2
    exit 2
fi
if ! curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf \
    "${VLLM_ENDPOINT%/}/models" > /dev/null; then
    echo "Target unavailable: start server.sh first ($VLLM_ENDPOINT)." >&2
    exit 2
fi

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"

nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
    -m speculators.train \
    --training-recipe legacy \
    --loss-implementation legacy \
    --optimizer muon \
    --muon-parameter-policy "$MUON_PARAMETER_POLICY" \
    --muon-lr "$MUON_LR" --muon-weight-decay "$MUON_WEIGHT_DECAY" \
    --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$OUTPUT_DIR/tensorboard" --run-name "$RUN_NAME" \
    --no-resume-from-checkpoint \
    --seed "$SEED" --epochs "$EPOCHS" --logger "$LOGGER" \
    --total-seq-len "$SEQ_LENGTH" --speculator-type mmuse \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --full-attention-indices "${FULL_ATTENTION_INDICES[@]}" \
    --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-correction-head \
    --correction-output-mode logits \
    --correction-hidden-size "$CORRECTION_HIDDEN_SIZE" \
    --correction-rank "$CORRECTION_RANK" \
    --correction-lm-head-fusion \
    --correction-num-layers "$CORRECTION_NUM_LAYERS" \
    --correction-num-heads "$CORRECTION_NUM_HEADS" \
    --correction-gate-bias "$CORRECTION_GATE_BIAS" \
    --correction-hidden-aux-loss \
    --correction-hidden-aux-weight "$CORRECTION_HIDDEN_AUX_WEIGHT" \
    --correction-hidden-feedback \
    --selector-correction-feedback corrected \
    --correction-project-corrected-hidden \
    --correction-with-markov \
    --correction-markov-gate-bias "$CORRECTION_MARKOV_GATE_BIAS" \
    --no-correction-rollout-metrics \
    --no-correction-base-diagnostics \
    --dflash-context-residual \
    --dflash-block-position-embedding \
    --dflash-gated-layer-fusion \
    --dflash2-dynamic-conv \
    --dflash2-conv-kernel-size "$CONV_KERNEL_SIZE" \
    --dflash2-conv-group-size "$CONV_GROUP_SIZE" \
    --dflash2-candidate-selector \
    --dflash2-selector-rank "$SELECTOR_RANK" \
    --dflash2-selector-top-k "$SELECTOR_TOP_K" \
    --dflash2-selector-greedy \
    --dflash2-selector-loss-weight "$SELECTOR_LOSS_WEIGHT" \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha 1.0 --confidence-length-alpha 0.0 \
    --confidence-loss-weighting match-draft --no-confidence-detach-features \
    --first-error-focal-alpha 0.0 --adaptive-loss none \
    --no-ssal-curriculum --ssal-curriculum-start 0.1 --ssal-curriculum-end 0.6 \
    --on-missing generate --on-generate delete \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Started $RUN_NAME, PID $TRAIN_PID, NPUs $TRAIN_NPUS"
echo "Log: $LOG_FILE"
echo "TensorBoard: $OUTPUT_DIR/tensorboard"
echo "Stop: kill $TRAIN_PID"
echo "Run the three experiments one at a time; their training NPUs overlap."
