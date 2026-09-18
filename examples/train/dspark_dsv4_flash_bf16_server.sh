#!/usr/bin/env bash
# Experimental HS server. Run from the repository root, in the A3 vLLM image.
# Historical filename: BF16 describes exported HS, not all target weights.
# The installed backend must support MODEL's quantization format on this hardware.
set -euo pipefail
# Defaults use one target host and one separate trainer host.
MODEL="${MODEL:-/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
HS_PATH="${HS_PATH:-/mnt/nfs/dataset/tmp_hs}"
VLLM_NPUS="${VLLM_NPUS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
TP_SIZE="${TP_SIZE:-8}"
DP_SIZE="${DP_SIZE:-2}"  # Global DP. DP4 uses two target hosts; DP1/2 uses one.
DP_START_RANK="${DP_START_RANK:-0}"
TARGET_MASTER_IP="${TARGET_MASTER_IP:-80.48.17.186}"
TARGET_WORKER_IP="${TARGET_WORKER_IP:-80.48.17.187}"
DP_ADDRESS="${DP_ADDRESS:-}"
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
VLLM_HOST="${VLLM_HOST:-$TARGET_MASTER_IP}"
VLLM_PORT="${VLLM_PORT:-8001}"
# User-provided topology: head .186, headless worker .187, same NIC on both.
# Override TARGET_LOCAL_IP/TARGET_IFNAME on other machines. Explicitly empty
# values retain backend interface selection on single-host DP1/2 only.
# Explicit per-backend environment settings take precedence over these defaults.
if [[ "$DP_START_RANK" == 2 ]]; then
  TARGET_LOCAL_IP="${TARGET_LOCAL_IP-$TARGET_WORKER_IP}"
else
  TARGET_LOCAL_IP="${TARGET_LOCAL_IP-$TARGET_MASTER_IP}"
fi
TARGET_IFNAME="${TARGET_IFNAME-enp48s3u1u1}"
if [[ -n "$TARGET_LOCAL_IP" ]]; then
  export HCCL_IF_IP="${HCCL_IF_IP:-$TARGET_LOCAL_IP}"
fi
if [[ -n "$TARGET_IFNAME" ]]; then
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$TARGET_IFNAME}"
  export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$TARGET_IFNAME}"
  export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-$TARGET_IFNAME}"
fi
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"
DSV4_MANIFEST_TIMEOUT="${DSV4_MANIFEST_TIMEOUT:-300}"
export DSV4_EVAL="${DSV4_EVAL:-0}"
DSV4_EXECUTION_MODE="${DSV4_EXECUTION_MODE:-eager}"
DSV4_ASYNC_SCHEDULING="${DSV4_ASYNC_SCHEDULING:-0}"
case "$DSV4_EXECUTION_MODE" in
  eager|full-decode-only) ;;
  *) printf '%s\n' 'DSV4_EXECUTION_MODE must be eager or full-decode-only.' >&2; exit 2 ;;
esac
case "$DSV4_ASYNC_SCHEDULING" in
  0) target_scheduling_args=(--no-async-scheduling) ;;
  1) target_scheduling_args=(--async-scheduling) ;;
  *) printf '%s\n' 'DSV4_ASYNC_SCHEDULING must be 0 or 1.' >&2; exit 2 ;;
esac
case "$DP_SIZE" in
  1|2) DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-$DP_SIZE}" ;;
  4) DP_SIZE_LOCAL="${DP_SIZE_LOCAL:-2}" ;;
  *) printf '%s\n' 'DP_SIZE must be 1 or 2 (one target host), or 4 (two target hosts).' >&2; exit 2 ;;
esac
target_parallel_args=()
target_http_args=(--enable-tokenizer-info-endpoint --host "$VLLM_HOST" --port "$VLLM_PORT")
HEADLESS=0
if [[ "$DP_SIZE" == 4 ]]; then
  DP_ADDRESS="${DP_ADDRESS:-$TARGET_MASTER_IP}"
  if [[ "$DP_SIZE_LOCAL" != 2 || ( "$DP_START_RANK" != 0 && "$DP_START_RANK" != 2 ) ]]; then
    echo "DP_SIZE=4 requires DP_SIZE_LOCAL=2 and DP_START_RANK=0 or 2." >&2
    exit 2
  fi
  if [[ -z "$DP_ADDRESS" ]]; then
    echo "DP_SIZE=4 requires DP_ADDRESS set to the first target host's reachable IP." >&2
    exit 2
  fi
  if [[ ! "$DP_RPC_PORT" =~ ^[1-9][0-9]{0,4}$ ]] || (( DP_RPC_PORT > 65535 )); then
    echo "DP_RPC_PORT must be an integer from 1 to 65535." >&2
    exit 2
  fi
  if [[ "$HS_PATH" != /* ]]; then
    echo "DP_SIZE=4 requires the same shared absolute HS_PATH on all three hosts." >&2
    exit 2
  fi
  if [[ -z "${HCCL_IF_IP:-}" || -z "${GLOO_SOCKET_IFNAME:-}" ||
        -z "${TP_SOCKET_IFNAME:-}" || -z "${HCCL_SOCKET_IFNAME:-}" ]]; then
    echo "DP_SIZE=4 requires this host's TARGET_LOCAL_IP and TARGET_IFNAME, or all four explicit HCCL/GLOO/TP network settings." >&2
    exit 2
  fi
  target_parallel_args=(--data-parallel-backend mp --data-parallel-address "$DP_ADDRESS"
    --data-parallel-rpc-port "$DP_RPC_PORT" --data-parallel-start-rank "$DP_START_RANK")
  if [[ "$DP_START_RANK" == 2 ]]; then
    HEADLESS=1
    target_parallel_args+=(--headless)
    target_http_args=()
  fi
elif [[ "$DP_SIZE_LOCAL" != "$DP_SIZE" || "$DP_START_RANK" != 0 || -n "$DP_ADDRESS" ]]; then
  echo "Single-host DP1/2 requires DP_SIZE_LOCAL=DP_SIZE, DP_START_RANK=0, and no DP_ADDRESS." >&2
  exit 2
fi
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

if [[ ! "$VLLM_STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "VLLM_STARTUP_TIMEOUT must be a positive number of seconds." >&2
  exit 2
fi
if [[ ! "$DSV4_MANIFEST_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
  echo "DSV4_MANIFEST_TIMEOUT must be a positive integer number of seconds." >&2
  exit 2
fi
printf 'DSV4 HS execution mode: %s; async scheduling: %s\n' \
  "$DSV4_EXECUTION_MODE" "$DSV4_ASYNC_SCHEDULING"
printf 'DSV4 HS topology: TP=%s, global DP=%s, local DP=%s, start rank=%s, headless=%s\n' \
  "$TP_SIZE" "$DP_SIZE" "$DP_SIZE_LOCAL" "$DP_START_RANK" "$HEADLESS"
if [[ "$DP_SIZE" == 4 ]]; then
  printf 'DSV4 target network: master=%s:%s, local HCCL IP=%s, GLOO=%s, TP=%s, HCCL=%s\n' \
    "$DP_ADDRESS" "$DP_RPC_PORT" "$HCCL_IF_IP" \
    "$GLOO_SOCKET_IFNAME" "$TP_SOCKET_IFNAME" "$HCCL_SOCKET_IFNAME"
fi
command -v setsid >/dev/null
if [[ "$HEADLESS" == 0 ]]; then command -v curl >/dev/null; fi
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
if [[ "$HEADLESS" == 0 ]] && curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "$READY_URL" >/dev/null 2>&1; then
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
  -u VLLM_DP_MASTER_IP -u VLLM_DP_MASTER_PORT \
  ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" \
  python scripts/launch_vllm.py "$MODEL" \
    --dsv4 \
    --dsv4-execution-mode "$DSV4_EXECUTION_MODE" \
    --dsv4-manifest-timeout "$DSV4_MANIFEST_TIMEOUT" \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids 1 11 21 30 40 \
    -- \
    "${target_quantization_args[@]}" \
    "${target_eval_args[@]}" \
    "${target_scheduling_args[@]}" \
    --tensor-parallel-size "$TP_SIZE" \
    --data-parallel-size "$DP_SIZE" \
    --data-parallel-size-local "$DP_SIZE_LOCAL" \
    "${target_parallel_args[@]}" \
    --distributed-executor-backend mp \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    "${target_http_args[@]}" \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --block-size 128 \
    --additional-config '{"enable_flashcomm1": false, "enable_dsa_cp": false}' &
VLLM_PID=$!
if [[ "$HEADLESS" == 1 ]]; then
  echo "Headless vLLM process launched (PID $VLLM_PID); this is NOT cluster readiness."
  echo "Wait for the first target host's HTTP service and run an HS probe before training."
  echo "Press Ctrl+C to stop only this host's owned vLLM process group."
  wait "$VLLM_PID"
  exit 0
fi
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
