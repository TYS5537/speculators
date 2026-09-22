#!/usr/bin/env bash
# Single-machine DSV4 block eval: manage a local target and evaluator together.
# The Python entry point owns readiness, per-run isolation, and child cleanup.
# This is not a serving-throughput benchmark and has not been verified on A3.
set -euo pipefail

# Dataset selection: edit the comma-separated JSONL names/stems here.
# Environment overrides are supported; DATASETS="" evaluates all discovered files.
DATASETS="${DATASETS-gsm8k,math500}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:?Set VERIFIER_MODEL to the local DSV4-Flash checkpoint}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to the trained DSV4 DSpark checkpoint}"
: "${DATASETS_ROOT:?Set DATASETS_ROOT to a JSONL file or directory}"
: "${HS_PATH:?Set HS_PATH to a parent directory for isolated per-run HS}"
: "${VLLM_NPUS:?Set VLLM_NPUS to comma-separated physical target NPU IDs}"
: "${EVAL_NPU:?Set EVAL_NPU to one physical evaluation NPU ID}"

cmd=(
  python3 scripts/evaluate/run_dsv4_offline_eval.py
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --datasets-root "$DATASETS_ROOT"
  --hidden-states-path "$HS_PATH"
  --target-devices "$VLLM_NPUS"
  --eval-device "$EVAL_NPU"
  --verification-mode "${VERIFICATION_MODE:-block}"
  --port "${VLLM_PORT:-0}"
  --max-model-len "${DSV4_MAX_MODEL_LEN:-4096}"
  --startup-timeout "${STARTUP_TIMEOUT:-1800}"
  --shutdown-timeout "${SHUTDOWN_TIMEOUT:-30}"
  --target-request-timeout "${TARGET_REQUEST_TIMEOUT:-120}"
  --output-dir "${OUTPUT_DIR:-dspark_dsv4_single_eval}"
  --max-samples "${MAX_SAMPLES:-4}"
  --max-new-tokens "${MAX_NEW_TOKENS:-64}"
  --temperature "${TEMPERATURE:-0.0}"
  --seed "${SEED:-980406}"
  --enable-thinking "${ENABLE_THINKING:-false}"
  --raw-prompt-mode "${RAW_PROMPT_MODE:-auto}"
)
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
