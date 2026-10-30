#!/bin/bash
# Run from the repository checkout; no vLLM server is needed.
# DATASETS_ROOT must be the same JSONL file/directory used for DSpark evaluation.
# Edit RUN_DIR and OUTPUT_DIR below, or set DRAFT_MODEL to any saved epoch.
# The model architecture comes from that checkpoint; NUM_LAYERS/NUM_LOOPS do
# not change the manually selected checkpoint path.
set -euo pipefail

# Verified-prefix KV reuse is enabled; append --no-draft-kv-reuse for reference.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/hs_connectors/src:$REPO_ROOT:${PYTHONPATH:-}"

MODEL="${MODEL:-../../Qwen3-4B}"
NUM_LAYERS="${NUM_LAYERS:-1}"
NUM_LOOPS="${NUM_LOOPS:-1}"
RUN_DIR="./output/leagle_qwen3_4b_700k_5ep_d1_r1_fullvocab_5hs"
DRAFT_MODEL="${DRAFT_MODEL:-$RUN_DIR/checkpoints/checkpoint_best}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"
TEMPERATURE="${TEMPERATURE:-1}"  # 0: greedy; >0: draft/target sampling.
CANDIDATE_HEAD="${CANDIDATE_HEAD:-auto}"  # Follow checkpoint; 0 forces dense, 1 forces candidates.
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-}"  # Empty uses checkpoint value (normally 128).
LEAGLE_SPARSE_CANDIDATES="${LEAGLE_SPARSE_CANDIDATES:-0}"  # Requires a trained candidate head.
LEAGLE_STATIC_KV="${LEAGLE_STATIC_KV:-0}"
LEAGLE_REUSE_ROLLOUT_BUFFERS="${LEAGLE_REUSE_ROLLOUT_BUFFERS:-0}"  # Requires static KV.
LEAGLE_STATIC_TARGET_KV="${LEAGLE_STATIC_TARGET_KV:-0}"
LEAGLE_SELECTIVE_TARGET_HIDDEN_STATES="${LEAGLE_SELECTIVE_TARGET_HIDDEN_STATES:-0}"
LEAGLE_SHARE_OUTPUT_HEAD="${LEAGLE_SHARE_OUTPUT_HEAD:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
ASCEND_DEVICES="${ASCEND_DEVICES:-0,1,2,3,4,5,6,7}"
DATASETS="${DATASETS:-gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca,arena-hard-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_leagle_5hs_temp1}"
DATASETS_ROOT="${DATASETS_ROOT:-../../DeepSpec/eval_datasets}"

cmd=(
    python scripts/evaluate/leagle_offline_eval.py
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
case "$CANDIDATE_HEAD" in
    auto) ;;
    0)
        cmd+=(--no-candidate-head)
        if [[ -n "$CANDIDATE_TOP_K" ]]; then
            echo 'CANDIDATE_TOP_K cannot be used with CANDIDATE_HEAD=0.' >&2; exit 2
        fi
        ;;
    1)
        cmd+=(--candidate-head)
        ;;
    *) echo 'CANDIDATE_HEAD must be auto, 0 or 1.' >&2; exit 2 ;;
esac
if [[ -n "$CANDIDATE_TOP_K" ]]; then cmd+=(--candidate-top-k "$CANDIDATE_TOP_K"); fi
for setting in LEAGLE_SPARSE_CANDIDATES LEAGLE_STATIC_KV LEAGLE_REUSE_ROLLOUT_BUFFERS \
    LEAGLE_STATIC_TARGET_KV LEAGLE_SELECTIVE_TARGET_HIDDEN_STATES LEAGLE_SHARE_OUTPUT_HEAD \
    PERSISTENT_WORKERS; do
    flag="${setting,,}"
    flag="${flag//_/-}"
    case "${!setting}" in
        0) cmd+=("--no-$flag") ;;
        1) cmd+=("--$flag") ;;
        *) echo "$setting must be 0 or 1." >&2; exit 2 ;;
    esac
done
case "${CHECK_GREEDY:-0}" in
    0) ;;
    1) cmd+=(--check-greedy) ;;
    *) echo 'CHECK_GREEDY must be 0 or 1.' >&2; exit 2 ;;
esac

echo "LEAGLE evaluation: $DRAFT_MODEL, temperature=$TEMPERATURE, candidate_head=$CANDIDATE_HEAD"
echo "Results: $OUTPUT_DIR"
exec "${cmd[@]}" "$@"
