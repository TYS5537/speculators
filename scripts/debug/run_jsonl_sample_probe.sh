#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/debug:${PYTHONPATH:-}"

: "${VERIFIER_MODEL:?set VERIFIER_MODEL to the verifier model path or HF id}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to the DSpark checkpoint}"
: "${DATASET:?set DATASET to a JSONL benchmark file, e.g. aime24.jsonl}"
: "${SAMPLE_INDEX:=0}"
: "${MAX_NEW_TOKENS:=32}"
: "${TEMPERATURE:=0.0}"
: "${TOP_K:=5}"
: "${ENABLE_THINKING:=false}"
: "${RAW_PROMPT_MODE:=auto}"
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${SAMPLE_FROM_ANCHOR:=}"
: "${TRUST_REMOTE_CODE:=1}"
: "${SEED:=0}"

cmd=(
  python3 scripts/debug/jsonl_sample_probe.py
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --dataset "$DATASET"
  --sample-index "$SAMPLE_INDEX"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --enable-thinking "$ENABLE_THINKING"
  --raw-prompt-mode "$RAW_PROMPT_MODE"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --draft-attn-impl "$DRAFT_ATTN_IMPL"
  --seed "$SEED"
)

if [[ -n "$SAMPLE_FROM_ANCHOR" ]]; then
  cmd+=(--sample-from-anchor "$SAMPLE_FROM_ANCHOR")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

exec "${cmd[@]}"
