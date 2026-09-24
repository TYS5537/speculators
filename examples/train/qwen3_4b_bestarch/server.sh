#!/usr/bin/env bash
# One target-only hidden-state server can serve all three training experiments.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
configure_ascend_training_env
# The supplied server used queue mode 1; training keeps mode 2.
export TASK_QUEUE_ENABLE=1
SERVER_PROVENANCE_DIR="${SERVER_PROVENANCE_DIR:-$OUTPUT_ROOT/server/$(date +%Y%m%d_%H%M%S)_$$}"
server_command=(env "ASCEND_RT_VISIBLE_DEVICES=$VLLM_NPUS"
    python scripts/launch_vllm.py train "$MODEL"
    --target-layer-ids "${TARGET_LAYER_IDS[@]}"
    --provenance-dir "$SERVER_PROVENANCE_DIR"
    -- --port "$VLLM_PORT" "${VLLM_EXTRA_ARGS[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${server_command[@]}"
    printf '\n'
    exit 0
fi
if [[ ! "$SERVER_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SERVER_READY_TIMEOUT must be a positive number of seconds" >&2
    exit 2
fi
server_url="http://127.0.0.1:${VLLM_PORT}/v1/models"
if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$server_url" > /dev/null; then
    echo "A server already responds on port $VLLM_PORT; reuse it after checking model/layers." >&2
    exit 2
fi
"${server_command[@]}" &
VLLM_PID=$!
cleanup() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + SERVER_READY_TIMEOUT))
until curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$server_url" > /dev/null; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited before becoming ready; inspect its output above." >&2
        status=0
        wait "$VLLM_PID" || status=$?
        if [[ "$status" == 0 ]]; then status=1; fi
        exit "$status"
    fi
    if (( SECONDS >= deadline )); then
        echo "vLLM readiness timed out after $SERVER_READY_TIMEOUT seconds." >&2
        exit 1
    fi
    sleep 2
done
echo "Shared Qwen3-4B target ready on port $VLLM_PORT, NPUs $VLLM_NPUS."
echo "Keep this terminal running; Ctrl+C stops this server only."
wait "$VLLM_PID"
