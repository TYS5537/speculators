#!/usr/bin/env bash
# Single-machine DSV4 block eval: manage a local target and evaluator together.
# The Python entry point owns readiness, per-run isolation, and child cleanup.
# This is not a serving-throughput benchmark and has not been verified on A3.
set -euo pipefail

# Dataset selection: edit the comma-separated JSONL names/stems here.
# Environment overrides are supported; DATASETS="" evaluates all discovered files.
DATASETS="${DATASETS-gsm8k,math500}"
# Empty uses the Qwen evaluator's per-dataset caps; set a number to override.
: "${MAX_SAMPLES:=}"
: "${DSV4_BLOCK_OUTPUT:=auto}"
: "${DSV4_PROFILE:=0}"  # Synchronized diagnostics, not serving speed.
: "${DSV4_KV_REUSE:=0}"
: "${DRAFT_KV_REUSE:=0}"  # Draft-only; works with either target mode.
: "${DSV4_KV_CACHE_MB:=1024}"  # Host RAM per target process.
: "${DSV4_REPLAY_CACHE:=dsv4_greedy_traces}"
: "${DSV4_REPLAY_CACHE_TAG:=}"
: "${DSV4_REPLAY_AUDIT_SAMPLES:=0}"
# Target concurrency cap is 16 (this managed launcher creates one DP engine).
# Token budget stays separate: KV hits need only their new suffix; cold prefixes
# may queue. More concurrent sequences need more KV/activation memory.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
# Both packages use a src layout when running directly from the checkout.
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:?Set VERIFIER_MODEL to the local DSV4-Flash checkpoint}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the trained DSV4 DSpark checkpoint}"
: "${DATASETS_ROOT:?Set DATASETS_ROOT to a JSONL file or directory}"
: "${HS_PATH:?Set HS_PATH to a parent directory for isolated per-run HS}"
: "${VLLM_NPUS:?Set VLLM_NPUS to comma-separated physical target NPU IDs}"
# One ID keeps single-card evaluation; a list runs one draft worker per card.
# Example on a 16-device host: VLLM_NPUS=0,1,2,3,4,5,6,7 EVAL_NPU=8,9,10,11,12,13,14,15
: "${EVAL_NPU:?Set EVAL_NPU to comma-separated physical evaluation NPU IDs}"

cmd=(
  python3 scripts/evaluate/run_dsv4_offline_eval.py
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --datasets-root "$DATASETS_ROOT"
  --hidden-states-path "$HS_PATH"
  --target-devices "$VLLM_NPUS"
  --eval-device "$EVAL_NPU"
  --verification-mode "${VERIFICATION_MODE:-block}"
  --dsv4-block-output "$DSV4_BLOCK_OUTPUT"
  --dsv4-replay-cache "$DSV4_REPLAY_CACHE"
  --dsv4-replay-cache-tag "$DSV4_REPLAY_CACHE_TAG"
  --dsv4-replay-audit-samples "$DSV4_REPLAY_AUDIT_SAMPLES"
  --port "${VLLM_PORT:-0}"
  --max-model-len "${DSV4_MAX_MODEL_LEN:-4096}"
  --target-max-num-seqs "${MAX_NUM_SEQS:-16}"
  --target-max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-${DSV4_MAX_MODEL_LEN:-4096}}"
  --startup-timeout "${STARTUP_TIMEOUT:-1800}"
  --shutdown-timeout "${SHUTDOWN_TIMEOUT:-30}"
  --target-request-timeout "${TARGET_REQUEST_TIMEOUT:-120}"
  --output-dir "${OUTPUT_DIR:-dspark_dsv4_single_eval}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --temperature "${TEMPERATURE:-0.0}"
  --seed "${SEED:-980406}"
  --enable-thinking "${ENABLE_THINKING:-false}"
  --raw-prompt-mode "${RAW_PROMPT_MODE:-auto}"
)
case "$DSV4_PROFILE" in
  0) ;;
  1) cmd+=(--dsv4-profile) ;;
  *) printf '%s\n' 'DSV4_PROFILE must be 0 or 1.' >&2; exit 1 ;;
esac
case "$DSV4_KV_REUSE" in
  0) ;;
  1) cmd+=(--dsv4-kv-reuse --dsv4-kv-cache-mb "$DSV4_KV_CACHE_MB") ;;
  *) echo 'DSV4_KV_REUSE must be 0 or 1.' >&2; exit 1 ;;
esac
case "$DRAFT_KV_REUSE" in
  0) ;;
  1) cmd+=(--draft-kv-reuse) ;;
  *) echo 'DRAFT_KV_REUSE must be 0 or 1.' >&2; exit 1 ;;
esac
if [[ -n "${TP_SIZE:-}" ]]; then
  cmd+=(--target-tp-size "$TP_SIZE")
fi
if [[ -n "${TARGET_PYTHON:-}" ]]; then
  cmd+=(--target-python "$TARGET_PYTHON")
fi
if [[ -n "${EVAL_PYTHON:-}" ]]; then
  cmd+=(--eval-python "$EVAL_PYTHON")
fi
if [[ -n "${TARGET_QUANTIZATION:-}" ]]; then
  cmd+=(--target-quantization "$TARGET_QUANTIZATION")
fi
if [[ -n "${TARGET_MEMORY_UTILIZATION:-}" ]]; then
  cmd+=(--target-memory-utilization "$TARGET_MEMORY_UTILIZATION")
fi
if [[ -n "${DATASETS:-}" ]]; then
  cmd+=(--datasets "$DATASETS")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi
case "${ALLOW_SHARED_DEVICE:-0}" in
  0) ;;
  1) cmd+=(--allow-shared-device) ;;
  *) printf '%s\n' 'ALLOW_SHARED_DEVICE must be 0 or 1.' >&2; exit 1 ;;
esac
case "${KEEP_TARGET_HS:-0}" in
  0) ;;
  1) cmd+=(--keep-target-hs) ;;
  *) printf '%s\n' 'KEEP_TARGET_HS must be 0 or 1.' >&2; exit 1 ;;
esac
case "${SKIP_ARTIFACTS:-0}" in
  0) ;;
  1) cmd+=(--skip-artifacts) ;;
  *) printf '%s\n' 'SKIP_ARTIFACTS must be 0 or 1.' >&2; exit 1 ;;
esac
case "${DRY_RUN:-0}" in
  0) ;;
  1) cmd+=(--dry-run) ;;
  *) printf '%s\n' 'DRY_RUN must be 0 or 1.' >&2; exit 1 ;;
esac

exec env -u LOCAL_RANK -u RANK -u WORLD_SIZE "${cmd[@]}" "$@"
