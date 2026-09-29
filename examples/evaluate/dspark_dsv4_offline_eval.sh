#!/usr/bin/env bash
# Experimental DSV4 acceptance-length evaluation through an HS server.
# Defaults to block verification using a dedicated --dsv4-block-verify target.
# VERIFICATION_MODE=reference needs DSV4_BLOCK_VERIFY=0 and DSV4_EVAL=1 there.
# VERIFICATION_MODE=replay requires DSV4_GREEDY_REPLAY=1 on that block target.
# Only the dense draft and target IO weights are loaded on the evaluation device.
# Full-prefix target recomputation and file/API transfers are NOT serving speed.
# Multi-NPU progress is aggregated in one bar, or periodic snapshots under nohup.
set -euo pipefail

# Dataset selection: edit the comma-separated JSONL names/stems here.
# Environment overrides are supported; DATASETS="" evaluates all discovered files.
DATASETS="${DATASETS-gsm8k,math500,humaneval,mbpp,mt-bench}"
# Empty uses the Qwen evaluator's per-dataset caps; set a number to override.
: "${MAX_SAMPLES:=}"
# Auto compacts temperature=0 block results when the draft needs no target logits.
# Requires the updated block server; full + DSV4_PROFILE=0 uses the old protocol.
: "${DSV4_BLOCK_OUTPUT:=full}"
# Diagnostic mode synchronizes devices and writes timing.json (not serving speed).
: "${DSV4_PROFILE:=0}"
# Requires DSV4_KV_REUSE=1 on the dedicated block target too.
: "${DSV4_KV_REUSE:=0}"
# Draft-only context/KV cache; independent of target reuse, no server changes.
: "${DRAFT_KV_REUSE:=1}"
# VERIFICATION_MODE=replay needs a DSV4_GREEDY_REPLAY=1 target; reuse across drafts.
: "${DSV4_REPLAY_CACHE:=./dsv4_greedy_traces}"
: "${DSV4_REPLAY_CACHE_TAG:=stack-v1}"
: "${DSV4_REPLAY_AUDIT_SAMPLES:=0}"  # First N samples per worker also run live block.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# Both packages use a src layout when running directly from the checkout.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT:${PYTHONPATH:-}"

# Local copy of the target checkpoint; its directory may differ from the server's.
# Preserve shard timestamps when copying: they are part of the checkpoint signature.
: "${VERIFIER_MODEL:=/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
: "${DRAFT_MODEL:=output/dspark_dsv4_flash_bestArch/checkpoints/9/}"
: "${DATASETS_ROOT:=../DeepSpec/eval_datasets}"
# HTTP mode needs no shared HS mount. HS_PATH then holds temporary local downloads.
# Set HS_HTTP_ENDPOINT="" explicitly to use shared-file transport instead.
# Saved deployment token matches the sidecar; override it on both hosts to rotate.
# Never put the token in CLI arguments.
HS_HTTP_ENDPOINT="${HS_HTTP_ENDPOINT-http://80.48.17.187:8002}"
if [[ -n "$HS_HTTP_ENDPOINT" ]]; then
  : "${DSV4_HS_HTTP_TOKEN:=8d4f1c7a9e2b6f30c5a1d8e74b9c2f61a7e5d3c8b0f2496e1c7a4d8b5f2e9031}"
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
  --dsv4-verification-mode "${VERIFICATION_MODE:-block}"
  --dsv4-block-output "$DSV4_BLOCK_OUTPUT"
  --dsv4-replay-cache "$DSV4_REPLAY_CACHE"
  --dsv4-replay-cache-tag "$DSV4_REPLAY_CACHE_TAG"
  --dsv4-replay-audit-samples "$DSV4_REPLAY_AUDIT_SAMPLES"
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --datasets-root "$DATASETS_ROOT"
  --hidden-states-path "$HS_PATH"
  --vllm-endpoint "$VLLM_ENDPOINT"
  --dsv4-max-model-len "${DSV4_MAX_MODEL_LEN:-8192}"
  --target-request-timeout "${TARGET_REQUEST_TIMEOUT:-1200}"
  --output-dir "${OUTPUT_DIR:-dspark_dsv4_reference_eval}"
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}"
  --temperature "${TEMPERATURE:-0.0}"
  --seed "${SEED:-980406}"
  --enable-thinking "${ENABLE_THINKING:-false}"
  --raw-prompt-mode "${RAW_PROMPT_MODE:-auto}"
  --device npu:0
  --dtype bfloat16
  --draft-attn-impl sdpa
)
case "$DSV4_PROFILE" in
  0) ;;
  1) cmd+=(--dsv4-profile) ;;
  *) printf '%s\n' 'DSV4_PROFILE must be 0 or 1.' >&2; exit 1 ;;
esac
case "$DSV4_KV_REUSE" in
  0) ;;
  1) cmd+=(--dsv4-kv-reuse) ;;
  *) echo 'DSV4_KV_REUSE must be 0 or 1.' >&2; exit 1 ;;
esac
case "$DRAFT_KV_REUSE" in
  0) ;;
  1) cmd+=(--draft-kv-reuse) ;;
  *) echo 'DRAFT_KV_REUSE must be 0 or 1.' >&2; exit 1 ;;
esac
if [[ -n "$HS_HTTP_ENDPOINT" ]]; then
  cmd+=(--hs-http-endpoint "$HS_HTTP_ENDPOINT")
fi
if [[ "$EVAL_NPU" == *,* ]]; then
  cmd+=(--ascend-devices "$EVAL_NPU")
fi
if [[ -n "${DATASETS:-}" ]]; then
  cmd+=(--datasets "$DATASETS")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
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
