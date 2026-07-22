#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Required inputs.
: "${VERIFIER_MODEL:?set VERIFIER_MODEL to the verifier model path or HF id}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to the DSpark checkpoint}"
: "${DATASETS_ROOT:?set DATASETS_ROOT to a JSONL file or directory}"

# Dataset/output controls.
: "${DATASETS:=}"
: "${OUTPUT_DIR:=dspark_offline_eval}"
: "${MAX_SAMPLES:=}"

# Generation controls. TEMPERATURE=0.0 is greedy; nonzero sampling applies to
# both target distributions and draft proposals.
: "${MAX_NEW_TOKENS:=512}"
: "${TEMPERATURE:=0.0}"
: "${SEED:=0}"

# Prompt handling.
: "${ENABLE_THINKING:=false}"
: "${RAW_PROMPT_MODE:=auto}"

# Runtime controls. For multi-NPU evaluation set ASCEND_DEVICES and keep
# DEVICE=npu:0; each worker receives one visible device internally.
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${ASCEND_DEVICES:=}"
: "${SKIP_ARTIFACTS:=0}"
: "${TRUST_REMOTE_CODE:=1}"

cmd=(
  python3 scripts/evaluate/dspark_offline_eval.py
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --datasets-root "$DATASETS_ROOT"
  --output-dir "$OUTPUT_DIR"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --enable-thinking "$ENABLE_THINKING"
  --raw-prompt-mode "$RAW_PROMPT_MODE"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --draft-attn-impl "$DRAFT_ATTN_IMPL"
  --seed "$SEED"
)

if [[ -n "$DATASETS" ]]; then
  cmd+=(--datasets "$DATASETS")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi
if [[ -n "$ASCEND_DEVICES" ]]; then
  cmd+=(--ascend-devices "$ASCEND_DEVICES")
fi
if [[ "$SKIP_ARTIFACTS" == "1" ]]; then
  cmd+=(--skip-artifacts)
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

exec "${cmd[@]}"
