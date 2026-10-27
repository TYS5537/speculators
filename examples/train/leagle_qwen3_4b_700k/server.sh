#!/bin/bash
# LEAGLE Qwen3-4B target server for the prepared 700k / 5-epoch training run.
# Run from the repository root. Edit this file directly; no shared settings.
# Run server.sh and train.sh on the same machine, with matching model, layers,
# endpoint and hidden-state path. The file backend requires a shared filesystem.
# These target NPUs overlap the older bestarch servers; the paired trainer's
# NPUs 2-7 also overlap the older trainers. Run those jobs one at a time.
set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=34655
export GLOO_SOCKET_IFNAME=lo

# ============ Configuration ============
MODEL="${MODEL:-../../Qwen3-4B}"
NUM_LAYERS="${NUM_LAYERS:-1}"
NUM_LOOPS="${NUM_LOOPS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/leagle_qwen3_4b_700k_5ep_d${NUM_LAYERS}_r${NUM_LOOPS}_fullvocab_5hs}"
VLLM_PORT="${VLLM_PORT:-8003}"
TARGET_LAYER_IDS=(1 9 17 25 33)  # Auxiliary layers only; the launcher appends teacher slot 36.
VLLM_NPUS="${VLLM_NPUS:-0,1}"
HIDDEN_STATES_PATH="$OUTPUT_DIR/hidden_states"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
SERVER_PROVENANCE_DIR="$OUTPUT_DIR/server/$(date +%Y%m%d_%H%M%S)_$$"

# ============ Launch ============
if [[ ! "$NUM_LAYERS" =~ ^[1-9][0-9]*$ || ! "$NUM_LOOPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_LAYERS and NUM_LOOPS must be positive integers." >&2
    exit 2
fi
if [[ ! "$VLLM_NPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "VLLM_NPUS must be comma-separated device IDs." >&2
    exit 2
fi
IFS=',' read -r -a SERVER_DEVICES <<< "$VLLM_NPUS"
VLLM_EXTRA_ARGS=(--data-parallel-size "${#SERVER_DEVICES[@]}")
if [[ ! "$SERVER_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SERVER_READY_TIMEOUT must be a positive number of seconds." >&2
    exit 2
fi
SERVER_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$SERVER_URL" > /dev/null; then
    echo "A server already responds on port $VLLM_PORT; check its model/layers or choose an unused port." >&2
    exit 2
fi

mkdir -p "$HIDDEN_STATES_PATH"
HIDDEN_STATES_PATH="$(cd "$HIDDEN_STATES_PATH" && pwd -P)"

env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py train "$MODEL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" --include-last-layer \
    --hidden-states-backend file --hidden-states-path "$HIDDEN_STATES_PATH" \
    --provenance-dir "$SERVER_PROVENANCE_DIR" \
    -- --port "$VLLM_PORT" --dtype bfloat16 "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Waiting for vLLM server to be ready..."
DEADLINE=$((SECONDS + SERVER_READY_TIMEOUT))
until curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$SERVER_URL" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready; inspect its output above." >&2
        STATUS=0
        wait "$VLLM_PID" || STATUS=$?
        if [[ "$STATUS" == 0 ]]; then STATUS=1; fi
        exit "$STATUS"
    fi
    if (( SECONDS >= DEADLINE )); then
        echo "vLLM readiness timed out after $SERVER_READY_TIMEOUT seconds." >&2
        exit 1
    fi
    sleep 2
done
echo "LEAGLE Qwen3-4B target ready on port $VLLM_PORT, NPUs $VLLM_NPUS."
echo "Hidden states: $HIDDEN_STATES_PATH"
echo "Provenance: $SERVER_PROVENANCE_DIR"
echo "Keep this terminal running; Ctrl+C stops this server only."
wait "$VLLM_PID"
