#!/bin/bash
# Run from the repository checkout; no vLLM server is needed.
# DATASETS_ROOT must be the same JSONL file/directory used for DSpark evaluation.
# Edit RUN_DIR and OUTPUT_DIR below, or set DRAFT_MODEL to any saved epoch.
# The model architecture comes from that checkpoint; NUM_LAYERS does not
# change the manually selected checkpoint path.
# Verified-prefix KV reuse is enabled; append --no-draft-kv-reuse for reference.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT:${PYTHONPATH:-}"

MODEL="${MODEL:-../../Qwen3-4B}"
NUM_LAYERS="${NUM_LAYERS:-1}"
RUN_DIR="./output/eagle3_qwen3_4b_700k_5ep_d1_fullvocab"
DRAFT_MODEL="${DRAFT_MODEL:-$RUN_DIR/checkpoints/checkpoint_best}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"
TEMPERATURE="${TEMPERATURE:-1}"  # 0: greedy; >0: draft/target sampling.
ASCEND_DEVICES="${ASCEND_DEVICES:-0,1,2,3,4,5,6,7}"
DATASETS="${DATASETS:-gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca,arena-hard-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_eagle_temp1}"
DATASETS_ROOT="${DATASETS_ROOT:-../../DeepSpec/eval_datasets}"

cmd=(
    python scripts/evaluate/eagle3_offline_eval.py
    --verifier-model "$MODEL" --draft-model "$DRAFT_MODEL"
    --datasets-root "$DATASETS_ROOT" --output-dir "$OUTPUT_DIR"
    --spec-tokens "$SPEC_TOKENS" --temperature "$TEMPERATURE"
    --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --seed "${SEED:-980406}"
    --device "${DEVICE:-npu:0}" --dtype "${DTYPE:-bfloat16}"
    --draft-attn-impl sdpa --target-attn-impl sdpa
    --enable-thinking "${ENABLE_THINKING:-false}"
    --raw-prompt-mode "${RAW_PROMPT_MODE:-auto}"
)
if [[ -n "${DATASETS:-}" ]]; then cmd+=(--datasets "$DATASETS"); fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then cmd+=(--max-samples "$MAX_SAMPLES"); fi
if [[ -n "${ASCEND_DEVICES:-}" ]]; then cmd+=(--ascend-devices "$ASCEND_DEVICES"); fi
case "${CHECK_GREEDY:-0}" in
    0) ;;
    1) cmd+=(--check-greedy) ;;
    *) echo 'CHECK_GREEDY must be 0 or 1.' >&2; exit 2 ;;
esac

echo "EAGLE3 evaluation: $DRAFT_MODEL, temperature=$TEMPERATURE"
echo "Results: $OUTPUT_DIR"
exec "${cmd[@]}" "$@"
