#!/bin/bash
# LEAGLE, one physical layer, one shared loop, full attention.
# Recipe: examples/train/eagle3_qwen3_4b_700k/train.sh (repository preset, not paper).
# This script runs only this experiment and manages its own target server.
set -euo pipefail

SCRIPT_LOCATION="${BASH_SOURCE[0]}"
[[ "$SCRIPT_LOCATION" == */* ]] || SCRIPT_LOCATION="./$SCRIPT_LOCATION"
SCRIPT_DIR="$(cd -- "${SCRIPT_LOCATION%/*}" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/run_leagle.sh"
cd -- "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: bash leagle/qwen3_4b_700k/run_leagle.sh [--dry-run]

LEAGLE, one physical layer, one shared loop, full attention.
  --dry-run  Print commands without creating files or starting processes.

Edit the recipe below. MODEL, DATA_PATH, OUTPUT_DIR, VLLM_PORT,
VLLM_NPUS, TRAIN_NPUS and SERVER_READY_TIMEOUT can also be set in the environment.
Relative paths are resolved from the repository root. The run stays in the foreground;
Ctrl+C stops the trainer and target server. Do not start server.sh separately.
This script trains one checkpoint run; it does not evaluate real acceptance lengths.
EOF
}

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54
export no_proxy="$NO_PROXY"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=34655 GLOO_SOCKET_IFNAME=lo

# ============ EAGLE-3 recipe; independently configurable ============
MODEL="${MODEL:-../../Qwen3-4B}"
DATA_PATH="${DATA_PATH:-../../datasets/open_perfectblend_qwen3_4b_700k}"
OUTPUT_DIR="${OUTPUT_DIR:-./leagle/runs/leagle_qwen3_4b_700k_5ep_d1_r1_full}"
RUN_NAME=leagle_d1_r1_full
VLLM_PORT="${VLLM_PORT:-8003}"
VLLM_NPUS="${VLLM_NPUS:-0,1}"
TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"

SEQ_LENGTH=3072
EPOCHS=5
SEED=42
TRAIN_DATA_RATIO=0.9
LOGGER=tensorboard
NOISE_STD=0.05
LR=1e-4
MUON_LR=1e-4
WEIGHT_DECAY=0.01
MUON_WEIGHT_DECAY=0.1
SCHEDULER_TYPE=linear
WARMUP_RATIO=0.01
DRAFT_VOCAB_SIZE=32000
TOKEN_FREQ_PATH="$DATA_PATH/token_freq.pt"
TARGET_LAYER_IDS=(2 18 33)  # The target launcher also extracts teacher slot 36.
DRAFT_ATTN_IMPL=sdpa  # Attention backend; keep SDPA for these training runs.
TTT_STEPS=7
TTT_STEP_LOSS_DECAY=1.0
LOSS_FN=kl_div
LOSS_IMPLEMENTATION=eager  # KL loss implementation only; attention still uses SDPA.

if [[ ! "$VLLM_NPUS" =~ ^[0-9]+(,[0-9]+)*$ ||
      ! "$TRAIN_NPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "VLLM_NPUS and TRAIN_NPUS must be comma-separated device IDs." >&2
    exit 2
fi
IFS=',' read -r -a SERVER_DEVICES <<< "$VLLM_NPUS"
IFS=',' read -r -a TRAIN_DEVICES <<< "$TRAIN_NPUS"
declare -A SEEN_DEVICES=()
for device in "${SERVER_DEVICES[@]}" "${TRAIN_DEVICES[@]}"; do
    device=$((10#$device))
    if [[ -n "${SEEN_DEVICES[$device]:-}" ]]; then
        echo "Server and training devices must be distinct, without duplicates." >&2
        exit 2
    fi
    SEEN_DEVICES[$device]=1
done
if [[ ! "$VLLM_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( VLLM_PORT > 65535 )); then
    echo "VLLM_PORT must be between 1 and 65535." >&2
    exit 2
fi
if [[ ! "$SERVER_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SERVER_READY_TIMEOUT must be a positive number of seconds." >&2
    exit 2
fi
[[ "$OUTPUT_DIR" == /* ]] || OUTPUT_DIR="$REPO_ROOT/$OUTPUT_DIR"
VLLM_ENDPOINT="http://127.0.0.1:${VLLM_PORT}/v1"
SERVER_URL="$VLLM_ENDPOINT/models"
HIDDEN_STATES_PATH="$OUTPUT_DIR/hidden_states"

SERVER_CMD=(env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" TASK_QUEUE_ENABLE=1
    python scripts/launch_vllm.py train "$MODEL"
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" --include-last-layer
    --hidden-states-backend file --hidden-states-path "$HIDDEN_STATES_PATH"
    --provenance-dir "$OUTPUT_DIR/checkpoints"
    -- --port "$VLLM_PORT" --dtype bfloat16
    --data-parallel-size "${#SERVER_DEVICES[@]}")
TRAIN_CMD=(env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" TASK_QUEUE_ENABLE=2
    torchrun --standalone --nproc_per_node "${#TRAIN_DEVICES[@]}"
    -m speculators.train
    --training-recipe upstream --loss-implementation "$LOSS_IMPLEMENTATION"
    --optimizer muon --muon-parameter-policy upstream
    --lr "$LR" --muon-lr "$MUON_LR"
    --weight-decay "$WEIGHT_DECAY" --muon-weight-decay "$MUON_WEIGHT_DECAY"
    --scheduler-type "$SCHEDULER_TYPE" --scheduler-warmup-ratio "$WARMUP_RATIO"
    --verifier-name-or-path "$MODEL" --data-path "$DATA_PATH"
    --hidden-states-backend file --hidden-states-path "$HIDDEN_STATES_PATH"
    --vllm-endpoint "$VLLM_ENDPOINT" --save-path "$OUTPUT_DIR/checkpoints"
    --log-dir "$OUTPUT_DIR/tensorboard" --run-name "$RUN_NAME"
    --no-resume-from-checkpoint --save-best
    --seed "$SEED" --epochs "$EPOCHS" --logger "$LOGGER"
    --total-seq-len "$SEQ_LENGTH" --train-data-ratio "$TRAIN_DATA_RATIO"
    --hidden-states-dtype bfloat16 --noise-std "$NOISE_STD"
    --speculator-type leagle --draft-arch qwen3 --num-layers 1
    --num-loops 1 --no-norm-before-fc
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" --token-freq-path "$TOKEN_FREQ_PATH"
    --target-layer-ids "${TARGET_LAYER_IDS[@]}"
    --draft-attn-impl "$DRAFT_ATTN_IMPL" --full-attention-indices 0
    --norm-output --no-embed-requires-grad
    --ttt-steps "$TTT_STEPS" --ttt-step-loss-decay "$TTT_STEP_LOSS_DECAY"
    --loss-fn "$LOSS_FN" --on-missing generate --on-generate delete)

print_command() { printf '%q ' "$@"; printf '\n'; }

if [[ "$DRY_RUN" == true ]]; then
    echo "[$RUN_NAME]"
    print_command "${SERVER_CMD[@]}"
    print_command "${TRAIN_CMD[@]}"
    exit 0
fi

# Validate this experiment before starting the target service.
if [[ ! -d "$MODEL" || ! -d "$DATA_PATH" ]]; then
    echo "Model or prepared Arrow data not found; check MODEL and DATA_PATH." >&2
    exit 2
fi
if [[ ! -f "$TOKEN_FREQ_PATH" ]] &&
   [[ ! -f "$DATA_PATH/d2t-${DRAFT_VOCAB_SIZE}.npy" ||
      ! -f "$DATA_PATH/t2d-${DRAFT_VOCAB_SIZE}.npy" ]]; then
    echo "Need token_freq.pt or paired size-keyed vocabulary mappings in DATA_PATH." >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Run directory already exists: $OUTPUT_DIR; choose a fresh OUTPUT_DIR." >&2
    exit 2
fi
for executable in python torchrun curl; do
    command -v "$executable" > /dev/null || { echo "Missing executable: $executable" >&2; exit 2; }
done

SERVER_PID=""
TRAIN_PID=""
cleanup() {
    local pid
    for pid in "$TRAIN_PID" "$SERVER_PID"; do
        if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
    done
    for pid in "$TRAIN_PID" "$SERVER_PID"; do
        if [[ -n "$pid" ]]; then wait "$pid" 2>/dev/null || true; fi
    done
    TRAIN_PID=""
    SERVER_PID=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$SERVER_URL" > /dev/null; then
    echo "Port $VLLM_PORT already serves a model; choose an unused VLLM_PORT." >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR/logs" "$HIDDEN_STATES_PATH"
cp -- "$SCRIPT_PATH" "$OUTPUT_DIR/run_leagle.sh"
{
    print_command "${SERVER_CMD[@]}"
    print_command "${TRAIN_CMD[@]}"
} > "$OUTPUT_DIR/launch_commands.txt"
echo "[$RUN_NAME] Starting target server; log: $OUTPUT_DIR/logs/server.log"
"${SERVER_CMD[@]}" > "$OUTPUT_DIR/logs/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" > "$OUTPUT_DIR/logs/server.pid"
DEADLINE=$((SECONDS + SERVER_READY_TIMEOUT))
until curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$SERVER_URL" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Target server exited; inspect $OUTPUT_DIR/logs/server.log" >&2
        exit 1
    fi
    if (( SECONDS >= DEADLINE )); then
        echo "Target readiness timed out; inspect $OUTPUT_DIR/logs/server.log" >&2
        exit 1
    fi
    sleep 2
done
echo "[$RUN_NAME] Training; log: $OUTPUT_DIR/logs/train.log"
"${TRAIN_CMD[@]}" > "$OUTPUT_DIR/logs/train.log" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$OUTPUT_DIR/logs/train.pid"
if wait "$TRAIN_PID"; then
    TRAIN_PID=""
else
    STATUS=$?
    TRAIN_PID=""
    echo "Training failed ($STATUS); inspect $OUTPUT_DIR/logs/train.log" >&2
    exit "$STATUS"
fi
cleanup
echo "[$RUN_NAME] Finished; checkpoints: $OUTPUT_DIR/checkpoints"
