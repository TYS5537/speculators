#!/usr/bin/env bash
# Experimental HS server. Run from the repository root, in the A3 vLLM image.
# Historical filename: BF16 describes exported HS, not all target weights.
# The installed backend must support MODEL's quantization format on this hardware.
set -euo pipefail
# These defaults are for two hosts: target here, trainer on the other host.
MODEL="${MODEL:-/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
HS_PATH="${HS_PATH:-/mnt/nfs/dataset/tmp_hs}"
VLLM_NPUS="${VLLM_NPUS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-2}"  # Local DP; visible device count must equal TP*DP.
VLLM_HOST="${VLLM_HOST:-80.48.17.187}"
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
export DSV4_EVAL="${DSV4_EVAL:-0}"
case "$DP_SIZE" in
  1|2) ;;
  *) printf '%s\n' 'DP_SIZE must be 1 or 2 (single-host HS service).' >&2; exit 2 ;;
esac
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

if [[ ! "$VLLM_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "VLLM_STARTUP_TIMEOUT must be a positive number of seconds." >&2
  exit 2
fi
command -v setsid >/dev/null
command -v curl >/dev/null
python scripts/check_dsv4_checkpoint.py "$MODEL"
target_quantization_args=()
if [[ -n "${TARGET_QUANTIZATION:-}" ]]; then
  target_quantization_args=(--quantization "$TARGET_QUANTIZATION")
fi
target_eval_args=()
case "${DSV4_EVAL:-0}" in
  0) ;;
  1)
    # The reference evaluator needs the full, unprocessed target distribution.
    target_eval_args=(--max-logprobs 129280 --logprobs-mode raw_logprobs --generation-config vllm)
    ;;
  *)
    printf '%s\n' 'DSV4_EVAL must be 0 (training HS) or 1 (reference evaluation).' >&2
    exit 1
    ;;
esac
READY_HOST="$VLLM_HOST"
case "$READY_HOST" in
  0.0.0.0) READY_HOST=127.0.0.1 ;;
  ::) READY_HOST='[::1]' ;;
esac
READY_URL="http://${READY_HOST}:${VLLM_PORT}/v1/models"
if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$READY_URL" >/dev/null 2>&1; then
  echo "A server already responds at $READY_URL; refusing to reuse it." >&2
  exit 1
fi

VLLM_PID=""
cleanup() {
  [[ -n "$VLLM_PID" ]] || return 0
  echo "Stopping this vLLM process group ($VLLM_PID)..."
  # setsid owns a new group. Never kill by name or touch another server.
  kill -TERM -- "-$VLLM_PID" 2>/dev/null || true
  local deadline=$((SECONDS + 30))
  while kill -0 -- "-$VLLM_PID" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 -- "-$VLLM_PID" 2>/dev/null; then
    kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
  fi
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
set +m  # No job control: setsid keeps $! as the session/process-group leader.
setsid env -u LOCAL_RANK -u RANK -u WORLD_SIZE \
  -u VLLM_DP_SIZE -u VLLM_DP_RANK -u VLLM_DP_RANK_LOCAL \
  ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" \
  python scripts/launch_vllm.py "$MODEL" \
    --dsv4 \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids 1 11 21 30 40 \
    -- \
    "${target_quantization_args[@]}" \
    "${target_eval_args[@]}" \
    --tensor-parallel-size "$TP_SIZE" \
    --data-parallel-size "$DP_SIZE" \
    --data-parallel-size-local "$DP_SIZE" \
    --distributed-executor-backend mp \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --enable-tokenizer-info-endpoint \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --block-size 128 \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --additional-config '{"enable_flashcomm1": false, "enable_dsa_cp": false}' &
VLLM_PID=$!
echo "Waiting for vLLM (PID $VLLM_PID) at $READY_URL..."
deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
while true; do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    status=0
    wait "$VLLM_PID" || status=$?
    echo "vLLM exited before becoming ready (status $status)." >&2
    (( status != 0 )) || status=1
    exit "$status"
  fi
  if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$READY_URL" >/dev/null 2>&1; then
    break
  fi
  if (( SECONDS >= deadline )); then
    echo "vLLM readiness timed out after $VLLM_STARTUP_TIMEOUT seconds." >&2
    exit 1
  fi
  sleep 5
done
echo "vLLM server ready. Press Ctrl+C to stop it."
wait "$VLLM_PID"
