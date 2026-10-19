#!/bin/bash
# Qwen3-4B target machine: vLLM extraction plus authenticated HTTP hidden states.
# Run from the repository root. Edit configuration here; no shared settings.
set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export GLOO_SOCKET_IFNAME=lo

# ============ Configuration ============
MODEL="../../Qwen3-4B"  # Target-local copy; trainer copy must have identical bytes.
VLLM_PORT=8001
HS_HTTP_PORT=8002
HTTP_TOKEN="ff7b93b9f2e5be1e3eb1c5448b37bb15d158b7b31c1c64a2ae2e6dd5244f9083"  # Shared with the trainer; keep the same value on both hosts.
BIND_HOST="0.0.0.0"  # Use on a trusted LAN/VPN; both ports must reach the trainer.
TARGET_LAYER_IDS=(1 9 17 25 33)
VLLM_NPUS="0,1"
VLLM_EXTRA_ARGS=(--data-parallel-size 2)
SERVER_READY_TIMEOUT=600
SERVER_PROVENANCE_DIR="./output/qwen3_4b_bestarch/server_http/$(date +%Y%m%d_%H%M%S)_$$"
HS_PATH="$SERVER_PROVENANCE_DIR/hidden_states"

# ============ Launch ============
if [[ ${#HTTP_TOKEN} -lt 32 ]]; then
    echo "Set the same HTTP_TOKEN in both HTTP scripts (at least 32 characters)." >&2
    exit 2
fi
export SPECULATORS_HS_HTTP_TOKEN="$HTTP_TOKEN"
if [[ ! "$SERVER_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SERVER_READY_TIMEOUT must be a positive number of seconds." >&2
    exit 2
fi
SERVER_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$SERVER_URL" > /dev/null; then
    echo "A server already responds on port $VLLM_PORT; choose an unused port." >&2
    exit 2
fi

mkdir -p "$HS_PATH"
HS_PATH="$(cd "$HS_PATH" && pwd)"
python -m speculators.remote_http prepare \
    --model "$MODEL" --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --hidden-states-path "$HS_PATH"
cp "$HS_PATH/qwen3_hs_http.json" "$SERVER_PROVENANCE_DIR/"

VLLM_PID=""
HS_HTTP_PID=""
cleanup() {
    if [[ -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    if [[ -n "$HS_HTTP_PID" ]]; then
        kill "$HS_HTTP_PID" 2>/dev/null || true
        wait "$HS_HTTP_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python -m speculators.remote_http serve \
    --hidden-states-path "$HS_PATH" --host "$BIND_HOST" --port "$HS_HTTP_PORT" &
HS_HTTP_PID=$!

env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py train "$MODEL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" --include-last-layer \
    --hidden-states-backend file --hidden-states-path "$HS_PATH" \
    --provenance-dir "$SERVER_PROVENANCE_DIR" \
    -- --host "$BIND_HOST" --port "$VLLM_PORT" --dtype bfloat16 \
    --served-model-name qwen3-4b-http "${VLLM_EXTRA_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for the Qwen HTTP target..."
DEADLINE=$((SECONDS + SERVER_READY_TIMEOUT))
until curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$SERVER_URL" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null || ! kill -0 "$HS_HTTP_PID" 2>/dev/null; then
        echo "vLLM or the HS HTTP service exited before readiness; inspect output above." >&2
        exit 1
    fi
    if (( SECONDS >= DEADLINE )); then
        echo "Qwen HTTP readiness timed out after $SERVER_READY_TIMEOUT seconds." >&2
        exit 1
    fi
    sleep 2
done
python -m speculators.remote_http check \
    --model "$MODEL" --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --hidden-states-path "$SERVER_PROVENANCE_DIR/http_check" \
    --hs-http-endpoints "http://127.0.0.1:${HS_HTTP_PORT}"
echo "Qwen target ready: vLLM port $VLLM_PORT; hidden states port $HS_HTTP_PORT."
echo "Provenance: $SERVER_PROVENANCE_DIR"
echo "Keep this terminal running; Ctrl+C stops these two services."
# Stop the other child too if either service exits during training.
wait -n "$VLLM_PID" "$HS_HTTP_PID"
