#!/bin/bash
# Offline standalone DFlash2 evaluation on the same JSONL workload and metrics
# used by examples/evaluate/dspark_offline_jsonl.sh.
#
# Usage:
#   VERIFIER_MODEL=/path/to/target-or-hf-id \
#   DRAFT_MODEL=/path/to/dflash2-checkpoint \
#   DATASETS_ROOT=/path/to/jsonl_dir \
#   bash examples/evaluate/dflash2_offline_jsonl.sh
#
# ``accepted_draft_length`` excludes the anchor. ``acceptance_length`` includes
# the verifier bonus token. DFlash2 block_size=8/sample_from_anchor=False and
# DSpark block_size=7/sample_from_anchor=True therefore both propose 7 tokens.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:?set VERIFIER_MODEL to the target/verifier model path or HF id}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to the trained DFlash2 checkpoint}"
: "${DATASETS_ROOT:?set DATASETS_ROOT to a JSONL file or directory of JSONL files}"

: "${DATASETS:=}"
: "${OUTPUT_DIR:=dflash2_offline_eval}"
: "${MAX_SAMPLES:=}"

: "${MAX_NEW_TOKENS:=512}"
: "${TEMPERATURE:=0.0}"
: "${SEED:=980406}"

: "${ASCEND_DEVICES:=}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"
: "${MEASURE_BASE_SPEEDUP:=0}"
: "${THROUGHPUT_WARMUP_SAMPLES:=1}"

: "${ENABLE_THINKING:=false}"
: "${RAW_PROMPT_MODE:=auto}"

cmd=(
  python3 scripts/evaluate/dflash2_offline_eval.py
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
if [[ "$MEASURE_BASE_SPEEDUP" == "1" ]]; then
  cmd+=(
    --measure-base-speedup
    --throughput-warmup-samples "$THROUGHPUT_WARMUP_SAMPLES"
  )
fi

exec "${cmd[@]}"
