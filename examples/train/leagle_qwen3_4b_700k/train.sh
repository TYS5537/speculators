#!/bin/bash
# LEAGLE on the existing Qwen3-4B 700k prepared dataset, for five epochs.
# Run from the repository root after starting server.sh in this directory.
# Edit the settings here and in server.sh together; no shared settings file.
# The original dataset's token IDs and loss masks are reused without regeneration.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

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
PYTHON="${PYTHON:-python}"
NUM_LAYERS="${NUM_LAYERS:-1}"
NUM_LOOPS="${NUM_LOOPS:-1}"
DATA_PATH="${DATA_PATH:-../../datasets/open_perfectblend_qwen3_4b_700k}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/leagle_qwen3_4b_700k_5ep_d${NUM_LAYERS}_r${NUM_LOOPS}_fullvocab_5hs}"
RUN_NAME="leagle_qwen3_4b_700k_5ep_d${NUM_LAYERS}_r${NUM_LOOPS}_fullvocab_5hs"
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
LOG_FREQ="${LOG_FREQ:-10}"  # Compute/log training diagnostics every N steps.
PROFILE_FREQ="${PROFILE_FREQ:-}"  # Optional synchronized phase timing; 0 disables it.
NOISE_STD="${NOISE_STD:-0.05}"

# LEAGLE: Qwen3 KV-conditioned layers, shared loops, and seven AR token steps.
# NUM_LAYERS * NUM_LOOPS is the effective decoder depth per token.
# Optimizer, loss and schedule follow qwen3_4b_bestarch/train_dspark.sh.
# Keep LEAGLE's depth/loops, vocabulary, feature layers and five-epoch budget.
LR="${LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SCHEDULER_TYPE="${SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
DRAFT_ARCH="qwen3"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-151936}"  # Qwen3-4B full vocabulary.
TOKEN_FREQ_PATH="$DATA_PATH/token_freq.pt"
# Qwen3-4B has 36 layers: the five auxiliary slots are 1, 9, 17, 25 and 33.
# server.sh appends final teacher slot 36; do not append it here.
TARGET_LAYER_IDS=(1 9 17 25 33)
DRAFT_ATTN_IMPL="sdpa"  # Attention backend; independent of LOSS_IMPLEMENTATION.
TTT_STEPS="${TTT_STEPS:-7}" # Token prediction steps; independent of NUM_LOOPS.
TTT_STEP_LOSS_DECAY="${TTT_STEP_LOSS_DECAY:-1.0}"
DEFAULT_LOSS_FN='{"ce": 0.1, "tv": 0.9}'
LOSS_FN="${LOSS_FN:-$DEFAULT_LOSS_FN}"
# Match DSpark's CE/TV numerical implementation; attention still uses SDPA.
LOSS_IMPLEMENTATION="${LOSS_IMPLEMENTATION:-legacy}"

# Optional candidate head, created and trained from scratch with the draft.
# Its detached forward-KL objective trains only the two retrieval matrices.
CANDIDATE_HEAD="${CANDIDATE_HEAD:-0}"
CANDIDATE_HEAD_RANK="${CANDIDATE_HEAD_RANK:-256}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-128}"
CANDIDATE_HEAD_LOSS_WEIGHT="${CANDIDATE_HEAD_LOSS_WEIGHT:-1.0}"
CANDIDATE_HEAD_CHUNK_SIZE="${CANDIDATE_HEAD_CHUNK_SIZE:-32}"

# Two optional masters: 0=off, 1=on, empty=use individual/saved settings.
# A nonempty master overrides its members, including individual CLI flags.
# Safe alone never enables chunking or sparse attention. Numerical=0 also
# disables their dependent caches and uses native candidate backward replay.
LEAGLE_SAFE_OPTIMIZATIONS="${LEAGLE_SAFE_OPTIMIZATIONS:-}"
LEAGLE_NUMERICAL_OPTIMIZATIONS="${LEAGLE_NUMERICAL_OPTIMIZATIONS:-}"
# Individual switches: empty means off for a new run and inherit on resume.
# To control members individually, the resolved master must be null/unset;
# on resume an empty environment value still inherits a saved master.
LEAGLE_REUSE_TRAINING_KV="${LEAGLE_REUSE_TRAINING_KV:-}"
LEAGLE_CHUNKED_LOSS="${LEAGLE_CHUNKED_LOSS:-}"
LEAGLE_SPARSE_ATTENTION="${LEAGLE_SPARSE_ATTENTION:-}"
LEAGLE_CANDIDATE_GRAD_ADDMM="${LEAGLE_CANDIDATE_GRAD_ADDMM:-}"
# Closer native gradients; extra forward/backward teacher projections, default off.
LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE="${LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE:-}"
LEAGLE_TEACHER_CACHE="${LEAGLE_TEACHER_CACHE:-}"
# Reuse frozen FP32 -> BF16 head casts within a batch; retains a BF16 weight copy.
LEAGLE_CACHE_FROZEN_HEAD_WEIGHTS="${LEAGLE_CACHE_FROZEN_HEAD_WEIGHTS:-}"
# Reuse teacher hit/miss indices during checkpoint replay (SAFE group).
LEAGLE_CACHE_TEACHER_INDICES="${LEAGLE_CACHE_TEACHER_INDICES:-}"
LEAGLE_MASKED_HEAD="${LEAGLE_MASKED_HEAD:-}"
LEAGLE_TEACHER_CACHE_SUPERVISED="${LEAGLE_TEACHER_CACHE_SUPERVISED:-}"
LEAGLE_SHARED_LOSS_NORMALIZATION="${LEAGLE_SHARED_LOSS_NORMALIZATION:-}"  # FP32 only; BF16 uses reference losses.
LEAGLE_STREAM_MAIN_LOSS="${LEAGLE_STREAM_MAIN_LOSS:-}"  # FP32 only; BF16 uses reference chunks.
LEAGLE_ALIGN_CANDIDATE_CHUNKS="${LEAGLE_ALIGN_CANDIDATE_CHUNKS:-}"
LEAGLE_SELECTIVE_CHECKPOINT="${LEAGLE_SELECTIVE_CHECKPOINT:-}"
LEAGLE_CACHE_ATTENTION_MASKS="${LEAGLE_CACHE_ATTENTION_MASKS:-}"
LEAGLE_ATTENTION_MASK_CACHE_MB="${LEAGLE_ATTENTION_MASK_CACHE_MB:-}"  # New run: 64 MiB/rank.
LEAGLE_FSDP_KEEP_FORWARD_PARAMS="${LEAGLE_FSDP_KEEP_FORWARD_PARAMS:-}"
LEAGLE_SHARE_OUTPUT_HEAD="${LEAGLE_SHARE_OUTPUT_HEAD:-}"
# DDP only: bucket views are SAFE; batched copies are NUMERICAL and version-gated.
LEAGLE_DDP_GRADIENT_AS_BUCKET_VIEW="${LEAGLE_DDP_GRADIENT_AS_BUCKET_VIEW:-}"
LEAGLE_DDP_BATCHED_GRAD_COPY="${LEAGLE_DDP_BATCHED_GRAD_COPY:-}"
# Empty preserves PyTorch's bucket policy, or the saved value on resume.
# NUMERICAL=0 resets this to the PyTorch default too.
LEAGLE_DDP_BUCKET_CAP_MB="${LEAGLE_DDP_BUCKET_CAP_MB:-}"
# Persistent teacher probabilities/IDs only; temporary chunk workspace is extra.
LEAGLE_TEACHER_CACHE_MB="${LEAGLE_TEACHER_CACHE_MB:-}"  # New run: 1024 MiB/rank.
# Empty uses 128 on a new run, or the saved size on resume.
LEAGLE_LOSS_CHUNK_SIZE="${LEAGLE_LOSS_CHUNK_SIZE:-}"
LEAGLE_ATTN_CHUNK_SIZE="${LEAGLE_ATTN_CHUNK_SIZE:-}"

FSDP_SHARD="${FSDP_SHARD:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"  # Print the resolved launch command without loading models.

# ============ Launch ============
for setting in FSDP_SHARD ACTIVATION_CHECKPOINTING RESUME DRY_RUN CANDIDATE_HEAD; do
    if [[ "${!setting}" != 0 && "${!setting}" != 1 ]]; then
        echo "$setting must be 0 or 1." >&2
        exit 2
    fi
done
if [[ ! "$LOG_FREQ" =~ ^[1-9][0-9]*$ ]]; then
    echo "LOG_FREQ must be a positive integer." >&2
    exit 2
fi
if [[ ! "$NUM_LAYERS" =~ ^[1-9][0-9]*$ || ! "$NUM_LOOPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_LAYERS and NUM_LOOPS must be positive integers." >&2
    exit 2
fi
# Qwen3-4B: full causal attention in every physical layer and every loop.
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

# Upstream keeps size-keyed vocabulary caches and hidden-state recovery behavior.
# The explicit optimizer/loss/scheduler settings above take precedence.
TRAIN_ARGS=(
    --training-recipe upstream --loss-implementation "$LOSS_IMPLEMENTATION" \
    --optimizer adamw --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
    --adam-beta1 "$ADAM_BETA1" --adam-beta2 "$ADAM_BETA2" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --scheduler-type "$SCHEDULER_TYPE" --scheduler-warmup-ratio "$WARMUP_RATIO" \
    --verifier-name-or-path "$MODEL" --data-path "$DATA_PATH" \
    --hidden-states-backend file --hidden-states-path "$HIDDEN_STATES_PATH" \
    --vllm-endpoint "$VLLM_ENDPOINT" --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$OUTPUT_DIR/tensorboard" --run-name "$RUN_NAME" \
    --no-resume-from-checkpoint --checkpoint-freq 1 \
    --seed "$SEED" --epochs "$EPOCHS" --logger "$LOGGER" \
    --log-freq "$LOG_FREQ" \
    --total-seq-len "$SEQ_LENGTH" --train-data-ratio "$TRAIN_DATA_RATIO" \
    --hidden-states-dtype bfloat16 --noise-std "$NOISE_STD" \
    --speculator-type leagle --num-layers "$NUM_LAYERS" --num-loops "$NUM_LOOPS" \
    --draft-arch "$DRAFT_ARCH" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" --token-freq-path "$TOKEN_FREQ_PATH" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --draft-attn-impl "$DRAFT_ATTN_IMPL" --full-attention-indices "${FULL_ATTENTION_INDICES[@]}" \
    --no-norm-before-fc --norm-output --no-embed-requires-grad \
    --ttt-steps "$TTT_STEPS" --ttt-step-loss-decay "$TTT_STEP_LOSS_DECAY" \
    --loss-fn "$LOSS_FN" --on-missing generate --on-generate delete \
)
if [[ "$RESUME" == 1 ]]; then
    # Architecture, optimizer, loss, schedule and memory settings come from the
    # original run. Endpoint and output paths may change when restarting.
    TRAIN_ARGS=(--config "$OUTPUT_DIR/checkpoints/run.yaml" --resume-from-checkpoint
        --log-freq "$LOG_FREQ"
        --no-save-best --checkpoint-freq 1
        --save-path "$OUTPUT_DIR/checkpoints" --log-dir "$OUTPUT_DIR/tensorboard"
        --vllm-endpoint "$VLLM_ENDPOINT" --hidden-states-path "$HIDDEN_STATES_PATH")
else
    if [[ "$CANDIDATE_HEAD" == 1 ]]; then
        TRAIN_ARGS+=(--candidate-head --candidate-head-rank "$CANDIDATE_HEAD_RANK"
            --candidate-head-top-k "$CANDIDATE_TOP_K"
            --candidate-head-loss-weight "$CANDIDATE_HEAD_LOSS_WEIGHT"
            --candidate-head-chunk-size "$CANDIDATE_HEAD_CHUNK_SIZE")
    fi
    if [[ "$FSDP_SHARD" == 1 ]]; then TRAIN_ARGS+=(--fsdp-shard); fi
    if [[ "$ACTIVATION_CHECKPOINTING" == 1 ]]; then TRAIN_ARGS+=(--activation-checkpointing); fi
fi
# Use this checkout and the selected Python environment for every worker.
if [[ -n "$PROFILE_FREQ" ]]; then
    if [[ ! "$PROFILE_FREQ" =~ ^[0-9]+$ ]]; then
        echo "PROFILE_FREQ must be empty or a nonnegative integer." >&2; exit 2
    fi
    TRAIN_ARGS+=(--profile-freq "$PROFILE_FREQ")
fi
for setting in LEAGLE_SAFE_OPTIMIZATIONS LEAGLE_NUMERICAL_OPTIMIZATIONS \
    LEAGLE_REUSE_TRAINING_KV LEAGLE_CHUNKED_LOSS LEAGLE_SPARSE_ATTENTION \
    LEAGLE_CANDIDATE_GRAD_ADDMM LEAGLE_TEACHER_CACHE LEAGLE_CACHE_FROZEN_HEAD_WEIGHTS \
    LEAGLE_CACHE_TEACHER_INDICES \
    LEAGLE_MASKED_HEAD \
    LEAGLE_TEACHER_CACHE_SUPERVISED LEAGLE_SHARED_LOSS_NORMALIZATION \
    LEAGLE_STREAM_MAIN_LOSS LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE \
    LEAGLE_ALIGN_CANDIDATE_CHUNKS LEAGLE_SELECTIVE_CHECKPOINT LEAGLE_CACHE_ATTENTION_MASKS \
    LEAGLE_FSDP_KEEP_FORWARD_PARAMS LEAGLE_SHARE_OUTPUT_HEAD \
    LEAGLE_DDP_GRADIENT_AS_BUCKET_VIEW LEAGLE_DDP_BATCHED_GRAD_COPY; do
    flag="${setting,,}"
    flag="${flag//_/-}"
    case "${!setting}" in
        '') ;;
        0) TRAIN_ARGS+=("--no-$flag") ;;
        1) TRAIN_ARGS+=("--$flag") ;;
        *) echo "$setting must be empty, 0 or 1." >&2; exit 2 ;;
    esac
done
for setting in LEAGLE_LOSS_CHUNK_SIZE LEAGLE_ATTN_CHUNK_SIZE LEAGLE_TEACHER_CACHE_MB \
    LEAGLE_ATTENTION_MASK_CACHE_MB LEAGLE_DDP_BUCKET_CAP_MB; do
    if [[ -n "${!setting}" ]]; then
        if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
            echo "$setting must be a positive integer." >&2; exit 2
        fi
        flag="${setting,,}"
        TRAIN_ARGS+=("--${flag//_/-}" "${!setting}")
    fi
done
CMD=(env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS"
    PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON" -m torch.distributed.run
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
