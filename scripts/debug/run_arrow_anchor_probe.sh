#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/scripts/debug:${PYTHONPATH:-}"

# Required inputs. DATA_PATH is the preprocessed Arrow dataset directory; hidden
# states are read from DATA_PATH/hidden_states unless HIDDEN_STATES_PATH is set.
: "${VERIFIER_MODEL:?set VERIFIER_MODEL to the verifier model path or HF id}"
: "${DRAFT_MODEL:?set DRAFT_MODEL to the DSpark checkpoint}"
: "${DATA_PATH:?set DATA_PATH to the preprocessed Arrow dataset directory}"
: "${HIDDEN_STATES_PATH:=}"

# Sample selection. Leave ANCHOR_POSITION empty to choose one valid loss_mask
# anchor randomly; set it to debug a specific token position.
: "${SAMPLE_START:=0}"
: "${RANDOM_SAMPLES:=1}"
: "${ANCHOR_POSITION:=}"
: "${TOP_K:=5}"
: "${CONTEXT_TOKENS:=16}"

# Runtime. Keep sample_from_anchor on the checkpoint config; overriding it from a
# wrapper script can hide the exact off-by-one issue this probe is meant to find.
: "${DEVICE:=npu:0}"
: "${DTYPE:=bfloat16}"
: "${DRAFT_ATTN_IMPL:=sdpa}"
: "${TRUST_REMOTE_CODE:=1}"
: "${SEED:=0}"

cmd=(
  python3 scripts/debug/arrow_anchor_probe.py
  --verifier-model "$VERIFIER_MODEL"
  --draft-model "$DRAFT_MODEL"
  --data-path "$DATA_PATH"
  --sample-start "$SAMPLE_START"
  --top-k "$TOP_K"
  --context-tokens "$CONTEXT_TOKENS"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --draft-attn-impl "$DRAFT_ATTN_IMPL"
  --seed "$SEED"
)

if [[ "$RANDOM_SAMPLES" == "1" ]]; then
  cmd+=(--random-samples)
fi
if [[ -n "$HIDDEN_STATES_PATH" ]]; then
  cmd+=(--hidden-states-path "$HIDDEN_STATES_PATH")
fi
if [[ -n "$ANCHOR_POSITION" ]]; then
  cmd+=(--anchor-position "$ANCHOR_POSITION")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

exec "${cmd[@]}"
