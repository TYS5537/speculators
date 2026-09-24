#!/bin/bash
# Shared Qwen3-4B target server for all three training scripts.
# Run from the repository root. Edit this file directly; no shared settings.
set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
export NO_PROXY=localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54
export no_proxy="$NO_PROXY"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=34655
export GLOO_SOCKET_IFNAME=lo

# ============ Configuration ============
MODEL="../../Qwen3-4B"
VLLM_PORT=8001
TARGET_LAYER_IDS=(1 9 17 25 33)
VLLM_NPUS="0,1"
VLLM_EXTRA_ARGS=(--data-parallel-size 2)
SERVER_READY_TIMEOUT=600
SERVER_PROVENANCE_DIR="./output/qwen3_4b_bestarch/server/$(date +%Y%m%d_%H%M%S)_$$"

# ============ Launch ============
if [[ ! "$SERVER_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SERVER_READY_TIMEOUT must be a positive number of seconds." >&2
    exit 2
fi
SERVER_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$SERVER_URL" > /dev/null; then
    echo "A server already responds on port $VLLM_PORT; reuse it after checking model/layers." >&2
    exit 2
fi

env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" python scripts/launch_vllm.py train "$MODEL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --provenance-dir "$SERVER_PROVENANCE_DIR" \
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}" &
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
echo "Shared Qwen3-4B target ready on port $VLLM_PORT, NPUs $VLLM_NPUS."
echo "Keep this terminal running; Ctrl+C stops this server only."
wait "$VLLM_PID"
