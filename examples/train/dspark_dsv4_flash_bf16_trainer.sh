#!/usr/bin/env bash
# Experimental DSV4 target adapter + user's Qwen3-4B recipe, corrGate=0.
# Run in the trainer environment from the repository root, after starting the HS server.
# Historical filename: target weights may be quantized; exported HS stay BF16.
set -euo pipefail
: "${MODEL:?Set MODEL to the same shared checkpoint used by the HS server}"
: "${DATA_PATH:?Set DATA_PATH to DSV4-tokenized Arrow data (NOT Qwen data)}"
: "${HS_PATH:?Set HS_PATH to the same shared HS directory used by the server}"
: "${TRAIN_NPUS:?Set TRAIN_NPUS to training-only device IDs}"
: "${NUM_TRAIN_NPUS:?Set NUM_TRAIN_NPUS to the number of visible training devices}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_dsv4_flash_corrGate0}"
TRAIN_ENTRY=(scripts/train.py)
case "${TRAINING_SMOKE:-0}" in
  0) ;;
  1)
    : "${SMOKE_PHASE:?Set SMOKE_PHASE to fresh or resume}"
    : "${SMOKE_REPORT_DIR:?Set SMOKE_REPORT_DIR to the isolated smoke report directory}"
    TRAIN_ENTRY=(scripts/check_dsv4_training.py
      --smoke-phase "$SMOKE_PHASE" --smoke-report-dir "$SMOKE_REPORT_DIR"
      --train-batches "${SMOKE_TRAIN_BATCHES:-2}" --val-batches "${SMOKE_VAL_BATCHES:-1}" --)
    ;;
  *) echo "TRAINING_SMOKE must be 0 or 1" >&2; exit 2 ;;
esac
export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0

# Explicit optimizer/scheduler pin the CURRENT local defaults. The user confirmed linear.
# The decoder keeps 32 Q / 8 KV heads, head_dim=128, FFN=9728, SWA=2048.
# Only hidden size / vocabulary change for target IO compatibility (4096 / 129280).
env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
  --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
  "${TRAIN_ENTRY[@]}" \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA_PATH" \
  --hidden-states-path "$HS_PATH" \
  --vllm-endpoint "${VLLM_ENDPOINT:-http://localhost:8001/v1}" \
  --save-path "$OUTPUT_DIR/checkpoints" \
  --target-hidden-state-format deepseek_v4_mean_hc_head \
  --draft-config examples/train/dsv4_flash_dense_config.json \
  --target-layer-ids 1 11 21 30 40 \
  --mask-token-id 128799 \
  --speculator-type dspark \
  --epochs 5 --lr 6e-5 --optimizer muon --scheduler-type linear \
  --logger tensorboard --total-seq-len 3072 \
  --block-size 7 --max-anchors 512 --draft-attn-impl sdpa \
  --markov-rank 256 --markov-head-type vanilla \
  --enable-correction-head --correction-output-mode logits \
  --correction-hidden-size 768 --correction-rank 256 \
  --no-correction-lm-head-fusion \
  --correction-num-layers 1 --correction-num-heads 8 --correction-gate-bias 0 \
  --correction-hidden-aux-loss --correction-hidden-aux-weight 0.1 \
  --correction-hidden-feedback --selector-correction-feedback corrected \
  --no-correction-project-corrected-hidden \
  --correction-with-markov --correction-markov-gate-bias -2.0 \
  --no-correction-rollout-metrics --no-correction-base-diagnostics \
  --dflash-context-residual --dflash-block-position-embedding --dflash-gated-layer-fusion \
  --dflash2-dynamic-conv --dflash2-conv-kernel-size 2 --dflash2-conv-group-size 16 \
  --dflash2-candidate-selector --dflash2-selector-rank 256 --dflash2-selector-top-k 16 \
  --dflash2-selector-greedy --dflash2-selector-loss-weight 0.1 \
  --enable-confidence-head --confidence-head-with-markov \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --confidence-head-alpha 1.0 --confidence-length-alpha 0.0 \
  --confidence-loss-weighting match-draft --no-confidence-detach-features \
  --first-error-focal-alpha 0.0 --adaptive-loss none \
  --no-ssal-curriculum --ssal-curriculum-start 0.1 --ssal-curriculum-end 0.6 \
  --on-missing generate --on-generate delete
