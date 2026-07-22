#!/bin/bash
# Offline DSpark evaluation on local JSONL datasets.
#
# This example is the public entry point for evaluating a trained DSpark draft
# model against a target/verifier model. DATASETS_ROOT can be either one JSONL
# file or a directory containing many JSONL files, such as:
#
#   datasets/eval/
#     aime24.jsonl
#     gsm8k.jsonl
#     math500.jsonl
#
# Usage:
#   VERIFIER_MODEL=/path/to/target-or-hf-id \
#   DRAFT_MODEL=/path/to/dspark-checkpoint \
#   DATASETS_ROOT=/path/to/jsonl_dir \
#   bash examples/evaluate/dspark_offline_jsonl.sh
#
# Optional:
#   DATASETS=aime24,gsm8k       # Only run selected JSONL files or stems.
#   MAX_SAMPLES=32              # Limit samples per dataset for smoke tests.
#   MAX_NEW_TOKENS=1024         # Max generated tokens per request.
#   TEMPERATURE=0.0             # 0.0 is greedy; >0 samples target and draft.
#   ASCEND_DEVICES=0,1,2,3      # Data-parallel evaluation across NPUs.
#   OUTPUT_DIR=eval_outputs/run1
#
# Internal DSpark layout details, including sample_from_anchor, are read from
# the checkpoint/config by the evaluator and intentionally are not exposed here.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Required inputs.
: "${VERIFIER_MODEL:?set VERIFIER_MODEL to the target/verifier model path or HF id}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to the trained DSpark checkpoint}"
: "${DATASETS_ROOT:?set DATASETS_ROOT to a JSONL file or directory of JSONL files}"

# Dataset/output controls.
: "${DATASETS:=}"
: "${OUTPUT_DIR:=dspark_offline_eval}"
: "${MAX_SAMPLES:=}"

# Generation controls.
: "${MAX_NEW_TOKENS:=512}"
: "${TEMPERATURE:=0.0}"
: "${SEED:=0}"

# Runtime controls. For multi-NPU evaluation set ASCEND_DEVICES and keep
# DEVICE=npu:0; each worker receives one visible device internally.
: "${ASCEND_DEVICES:=}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"

# Prompt handling. Use raw for preformatted prompts, chat_template for chat
# messages, or auto to infer from each JSONL row.
: "${ENABLE_THINKING:=false}"
: "${RAW_PROMPT_MODE:=auto}"

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
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

exec "${cmd[@]}"
