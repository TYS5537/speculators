#!/usr/bin/env bash
# Experimental DSV4 acceptance-length reference evaluation through an HS server.
# Start the target with DSV4_EVAL=1 using the matching --dsv4 HS bridge first.
# VERIFICATION_MODE=block requires a dedicated --dsv4-block-verify target instead.
# Only the dense draft and target IO weights are loaded on the evaluation device.
# Full-prefix target recomputation and file/API transfers are NOT serving speed.
set -euo pipefail

# Dataset selection: edit the comma-separated JSONL names/stems here.
# Environment overrides are supported; DATASETS="" evaluates all discovered files.
DATASETS="${DATASETS-gsm8k,math500}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:?Set VERIFIER_MODEL to the same shared DSV4 checkpoint as the HS server}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the trained DSV4 DSpark checkpoint}"
: "${DATASETS_ROOT:?Set DATASETS_ROOT to a JSONL file or directory of JSONL files}"
: "${HS_PATH:?Set HS_PATH to the shared HS server directory at the same absolute path}"
: "${VLLM_ENDPOINT:?Set VLLM_ENDPOINT to the trusted target service URL ending in /v1}"
# One draft worker per listed physical NPU; keep these separate from target devices.
: "${EVAL_NPU:?Set EVAL_NPU to comma-separated evaluation-only NPU IDs}"

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
  --max-samples "${MAX_SAMPLES:-4}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --temperature "${TEMPERATURE:-0.0}"
  --seed "${SEED:-980406}"
  --enable-thinking "${ENABLE_THINKING:-false}"
  --raw-prompt-mode "${RAW_PROMPT_MODE:-auto}"
  --device npu:0
  --dtype bfloat16
  --draft-attn-impl sdpa
)
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
