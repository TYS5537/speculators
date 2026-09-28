#!/usr/bin/env bash
# Experimental DSV4 acceptance-length reference evaluation through an HS server.
# Start the target with DSV4_EVAL=1 using the matching --dsv4 HS bridge first.
# VERIFICATION_MODE=block requires a dedicated --dsv4-block-verify target instead.
# Only the dense draft and target IO weights are loaded on the evaluation device.
# Full-prefix target recomputation and file/API transfers are NOT serving speed.
set -euo pipefail

# Dataset selection: edit the comma-separated JSONL names/stems here.
# Environment overrides are supported; DATASETS="" evaluates all discovered files.
DATASETS="${DATASETS-gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca,arena-hard-v2}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

# Local copy of the target checkpoint; its directory may differ from the server's.
# Preserve shard timestamps when copying: they are part of the checkpoint signature.
: "${VERIFIER_MODEL:=/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
: "${DRAFT_MODEL:=output/dspark_dsv4_flash_bestArch/checkpoints/9/}"
: "${DATASETS_ROOT:=../DeepSpec/eval_datasets}"
# HTTP mode needs no shared HS mount. HS_PATH then holds temporary local downloads.
# Export the same DSV4_HS_HTTP_TOKEN on both machines; never put it in CLI arguments.
HS_HTTP_ENDPOINT="${HS_HTTP_ENDPOINT:-}"
if [[ -n "$HS_HTTP_ENDPOINT" ]]; then
  : "${DSV4_HS_HTTP_TOKEN:?Export the HS sidecar bearer token}"
  export DSV4_HS_HTTP_TOKEN
  HS_PATH="${HS_PATH:-${OUTPUT_DIR:-dspark_dsv4_reference_eval}/target-hs-downloads}"
else
  : "${HS_PATH:=/mnt/nfs/dataset/tmp_hs}"
fi
: "${VLLM_ENDPOINT:=http://80.48.17.187:8001/v1}"
# One draft worker per listed physical NPU; keep these separate from target devices.
: "${EVAL_NPU:=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

cmd=(
  python3 scripts/evaluate/dspark_offline_eval.py
  --target-backend dsv4-vllm
  --dsv4-verification-mode "${VERIFICATION_MODE:-reference}"
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --datasets-root "$DATASETS_ROOT"
  --hidden-states-path "$HS_PATH"
  --vllm-endpoint "$VLLM_ENDPOINT"
  --dsv4-max-model-len "${DSV4_MAX_MODEL_LEN:-4096}"
  --target-request-timeout "${TARGET_REQUEST_TIMEOUT:-120}"
  --output-dir "${OUTPUT_DIR:-dspark_dsv4_reference_eval}"
  --max-samples "${MAX_SAMPLES:-500}"
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0.0}"
  --seed "${SEED:-980406}"
  --enable-thinking "${ENABLE_THINKING:-false}"
  --raw-prompt-mode "${RAW_PROMPT_MODE:-auto}"
  --device npu:0
  --dtype bfloat16
  --draft-attn-impl sdpa
)
if [[ -n "$HS_HTTP_ENDPOINT" ]]; then
  cmd+=(--hs-http-endpoint "$HS_HTTP_ENDPOINT")
fi
if [[ "$EVAL_NPU" == *,* ]]; then
  cmd+=(--ascend-devices "$EVAL_NPU")
fi
if [[ -n "${DATASETS:-}" ]]; then
  cmd+=(--datasets "$DATASETS")
fi
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
  cmd+=(--served-model-name "$SERVED_MODEL_NAME")
fi
case "${KEEP_TARGET_HS:-0}" in
  0) ;;
  1) cmd+=(--keep-target-hs) ;;
  *)
    printf '%s\n' 'KEEP_TARGET_HS must be 0 or 1.' >&2
    exit 1
    ;;
esac

exec env -u LOCAL_RANK -u RANK -u WORLD_SIZE \
  ASCEND_RT_VISIBLE_DEVICES="$EVAL_NPU" "${cmd[@]}"
