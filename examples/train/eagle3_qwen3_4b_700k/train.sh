#!/bin/bash
# EAGLE3 on the existing Qwen3-4B 700k prepared dataset, for five epochs.
# Run from the repository root after starting server.sh in this directory.
# Edit the settings here and in server.sh together; no shared settings file.
# The original dataset's token IDs and loss masks are reused without regeneration.
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
MODEL="${MODEL:-../../Qwen3-4B}"
NUM_LAYERS="${NUM_LAYERS:-1}"
DATA_PATH="${DATA_PATH:-../../datasets/open_perfectblend_qwen3_4b_700k}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/eagle3_qwen3_4b_700k_5ep_d${NUM_LAYERS}_full}"
RUN_NAME="eagle3_qwen3_4b_700k_5ep_d${NUM_LAYERS}_full"
VLLM_PORT="${VLLM_PORT:-8003}"
VLLM_ENDPOINT="http://127.0.0.1:${VLLM_PORT}/v1"
HIDDEN_STATES_PATH="$OUTPUT_DIR/hidden_states"
TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}"

# Keep the existing 700k data/packing budget, with the requested five epochs.
SEQ_LENGTH="${SEQ_LENGTH:-3072}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.9}"
LOGGER="${LOGGER:-tensorboard}"
NOISE_STD="${NOISE_STD:-0.05}"

# EAGLE online-example backbone: one layer, 32K vocabulary, and the same base
# LR for both optimizer groups. Seven TTT steps match DSpark's token horizon.
LR="${LR:-1e-4}"
MUON_LR="${MUON_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.1}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-linear}"
WARMUP_RATIO="${WARMUP_RATIO:-0.01}"
DRAFT_ARCH="llama"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
TOKEN_FREQ_PATH="$DATA_PATH/token_freq.pt"
# Qwen3-4B has 36 layers: the default auxiliary slots are 2, 18 and 33.
# server.sh appends final teacher slot 36; do not append it here.
TARGET_LAYER_IDS=(2 18 33)
DRAFT_ATTN_IMPL="sdpa"  # Attention backend; independent of LOSS_IMPLEMENTATION.
TTT_STEPS="${TTT_STEPS:-7}" # Token prediction steps; independent of any future inner loop depth.
TTT_STEP_LOSS_DECAY="${TTT_STEP_LOSS_DECAY:-1.0}"
LOSS_FN="${LOSS_FN:-kl_div}"
# Eager KL keeps this preset independent of the installed Triton/NPU loss stack.
# This changes the loss implementation, not the upstream recipe or KL objective.
LOSS_IMPLEMENTATION="${LOSS_IMPLEMENTATION:-eager}"

FSDP_SHARD="${FSDP_SHARD:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"  # Print the resolved launch command without loading models.

# ============ Launch ============
for setting in FSDP_SHARD ACTIVATION_CHECKPOINTING RESUME DRY_RUN; do
    if [[ "${!setting}" != 0 && "${!setting}" != 1 ]]; then
        echo "$setting must be 0 or 1." >&2
        exit 2
    fi
done
if [[ ! "$NUM_LAYERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_LAYERS must be a positive integer." >&2
    exit 2
fi
# Qwen3-4B: full causal attention in every physical draft layer.
FULL_ATTENTION_INDICES=()
for ((layer_idx=0; layer_idx<NUM_LAYERS; layer_idx++)); do
    FULL_ATTENTION_INDICES+=("$layer_idx")
done

if [[ ! "$TRAIN_NPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "TRAIN_NPUS must be comma-separated device IDs." >&2
    exit 2
fi
IFS=',' read -r -a TRAIN_DEVICES <<< "$TRAIN_NPUS"
NUM_TRAIN_NPUS="${#TRAIN_DEVICES[@]}"
if [[ "$RESUME" == 0 && "$DRY_RUN" == 0 ]] && [[ ! -d "$MODEL" || ! -d "$DATA_PATH" ]]; then
    echo "Model or prepared Arrow data not found; check MODEL and DATA_PATH from the repository root." >&2
    exit 2
fi
# Upstream uses size-keyed mappings. Do not silently fall back to full vocabulary
# when only another experiment's legacy t2d.npy/d2t.npy files are present.
if [[ "$RESUME" == 0 && "$DRY_RUN" == 0 && ! -f "$TOKEN_FREQ_PATH" ]] &&
   [[ ! -f "$DATA_PATH/d2t-${DRAFT_VOCAB_SIZE}.npy" ||
      ! -f "$DATA_PATH/t2d-${DRAFT_VOCAB_SIZE}.npy" ]]; then
    echo "Need $TOKEN_FREQ_PATH or paired d2t-${DRAFT_VOCAB_SIZE}.npy/t2d-${DRAFT_VOCAB_SIZE}.npy in DATA_PATH." >&2
    exit 2
fi
if [[ "$DRY_RUN" == 0 ]]; then
    if [[ -f "$OUTPUT_DIR/logs/train.pid" ]]; then
        read -r PREVIOUS_PID < "$OUTPUT_DIR/logs/train.pid" || true
        if [[ "${PREVIOUS_PID:-}" =~ ^[0-9]+$ ]] && kill -0 "$PREVIOUS_PID" 2>/dev/null; then
            echo "Trainer PID $PREVIOUS_PID is still running; refusing a second writer." >&2
            exit 2
        fi
    fi
    if [[ "$RESUME" == 1 ]]; then
        if [[ ! -f "$OUTPUT_DIR/checkpoints/run.yaml" ]]; then
            echo "RESUME=1 requires the original checkpoints/run.yaml." >&2
            exit 2
        fi
        if ! compgen -G "$OUTPUT_DIR/checkpoints/*/training_state.json" > /dev/null; then
            echo "No saved training progress in $OUTPUT_DIR/checkpoints." >&2
            exit 2
        fi
    elif [[ -e "$OUTPUT_DIR/checkpoints" || -e "$OUTPUT_DIR/logs/train.pid" ]]; then
        echo "Run directory already used: $OUTPUT_DIR; use RESUME=1 or a new OUTPUT_DIR." >&2
        exit 2
    fi
    if ! curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf \
        "${VLLM_ENDPOINT%/}/models" > /dev/null; then
        echo "Target unavailable: start the paired server.sh first ($VLLM_ENDPOINT)." >&2
        exit 2
    fi
fi

if [[ "$DRY_RUN" == 0 ]]; then
    mkdir -p "$HIDDEN_STATES_PATH"
    HIDDEN_STATES_PATH="$(cd "$HIDDEN_STATES_PATH" && pwd -P)"
fi

TRAIN_ARGS=(
    --training-recipe upstream --loss-implementation "$LOSS_IMPLEMENTATION" \
    --optimizer muon --muon-parameter-policy upstream \
    --lr "$LR" --muon-lr "$MUON_LR" \
    --weight-decay "$WEIGHT_DECAY" --muon-weight-decay "$MUON_WEIGHT_DECAY" \
    --scheduler-type "$SCHEDULER_TYPE" --scheduler-warmup-ratio "$WARMUP_RATIO" \
    --verifier-name-or-path "$MODEL" --data-path "$DATA_PATH" \
    --hidden-states-backend file --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "$VLLM_ENDPOINT" --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$OUTPUT_DIR/tensorboard" --run-name "$RUN_NAME" \
    --no-resume-from-checkpoint --checkpoint-freq 1 \
    --seed "$SEED" --epochs "$EPOCHS" --logger "$LOGGER" \
    --total-seq-len "$SEQ_LENGTH" --train-data-ratio "$TRAIN_DATA_RATIO" \
    --hidden-states-dtype bfloat16 --noise-std "$NOISE_STD" \
    --speculator-type eagle3 --num-layers "$NUM_LAYERS" --draft-arch "$DRAFT_ARCH" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" --token-freq-path "$TOKEN_FREQ_PATH" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" --full-attention-indices "${FULL_ATTENTION_INDICES[@]}" \
    --norm-before-residual --norm-before-fc --norm-output --no-embed-requires-grad \
    --ttt-steps "$TTT_STEPS" --ttt-step-loss-decay "$TTT_STEP_LOSS_DECAY" \
    --loss-fn "$LOSS_FN" --on-missing generate --on-generate delete \
)
if [[ "$RESUME" == 1 ]]; then
    # Architecture, optimizer, loss, schedule and memory settings come from the
    # original run. Endpoint and output paths may change when restarting.
    TRAIN_ARGS=(--config "$OUTPUT_DIR/checkpoints/run.yaml" --resume-from-checkpoint
        --no-save-best --checkpoint-freq 1
        --save-path "$OUTPUT_DIR/checkpoints" --log-dir "$OUTPUT_DIR/tensorboard"
        --vllm-endpoint "$VLLM_ENDPOINT" --hidden-states-path "$HIDDEN_STATES_PATH")
else
    if [[ "$FSDP_SHARD" == 1 ]]; then TRAIN_ARGS+=(--fsdp-shard); fi
    if [[ "$ACTIVATION_CHECKPOINTING" == 1 ]]; then TRAIN_ARGS+=(--activation-checkpointing); fi
fi
CMD=(env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" -m speculators.train "${TRAIN_ARGS[@]}")
if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_$$.log"
PID_FILE="$LOG_DIR/train.pid"
nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Started $RUN_NAME, PID $TRAIN_PID, NPUs $TRAIN_NPUS"
echo "Log: $LOG_FILE"
echo "TensorBoard: $OUTPUT_DIR/tensorboard"
echo "Checkpoints: $OUTPUT_DIR/checkpoints (all epochs retained; best validation loss: checkpoint_best)"
echo "Stop trainer: kill $TRAIN_PID; server.sh remains running independently."
echo "Run separately from other recipes using these NPUs; check the log for startup errors."
