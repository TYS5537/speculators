#!/usr/bin/env bash
# Experimental HS server. Run from the repository root, in the A3 vLLM image.
# Historical filename: BF16 describes exported HS, not all target weights.
# The installed backend must support MODEL's quantization format on this hardware.
set -euo pipefail
: "${MODEL:?Set MODEL to the local DSV4-Flash checkpoint directory}"
: "${HS_PATH:?Set HS_PATH to a new shared directory for DSV4 hidden states}"
: "${VLLM_NPUS:?Set VLLM_NPUS to target-only device IDs (do not overlap training)}"
: "${TP_SIZE:?Set TP_SIZE to the target tensor-parallel degree for your topology}"
: "${VLLM_HOST:?Set VLLM_HOST to the target host's trusted internal IP or 127.0.0.1}"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python scripts/check_dsv4_checkpoint.py "$MODEL"
target_quantization_args=()
if [[ -n "${TARGET_QUANTIZATION:-}" ]]; then
  target_quantization_args=(--quantization "$TARGET_QUANTIZATION")
fi
target_eval_args=()
case "${DSV4_EVAL:-0}" in
  0) ;;
  1)
    # The reference evaluator needs the full, unprocessed target distribution.
    target_eval_args=(--max-logprobs 129280 --logprobs-mode raw_logprobs --generation-config vllm)
    ;;
  *)
    printf '%s\n' 'DSV4_EVAL must be 0 (training HS) or 1 (reference evaluation).' >&2
    exit 1
    ;;
esac
env -u LOCAL_RANK -u RANK -u WORLD_SIZE ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" \
  python scripts/launch_vllm.py "$MODEL" \
    --dsv4 \
    --hidden-states-path "$HS_PATH" \
    --target-layer-ids 1 11 21 30 40 \
    -- \
    "${target_quantization_args[@]}" \
    "${target_eval_args[@]}" \
    --tensor-parallel-size "$TP_SIZE" \
    --data-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --enable-expert-parallel \
    --tokenizer-mode deepseek_v4 \
    --enable-tokenizer-info-endpoint \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --block-size 128 \
    --host "$VLLM_HOST" \
    --port "${VLLM_PORT:-8001}" \
    --additional-config '{"enable_flashcomm1": false, "enable_dsa_cp": false}'
