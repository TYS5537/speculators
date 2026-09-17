#!/usr/bin/env bash
# Experimental DSV4 target adapter + user's Qwen3-4B recipe, corrGate=0.
# Run in the trainer environment from the repository root, after starting the HS server.
# Historical filename: target weights may be quantized; exported HS stay BF16.
set -euo pipefail
# This script runs on a DIFFERENT host from the 16-device target server.
MODEL="${MODEL:-/mnt/nfs/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}"
DATA_PATH="${DATA_PATH:-/mnt/nfs/dataset/arrow_0730_77w_dedup}"
HS_PATH="${HS_PATH:-/mnt/nfs/dataset/tmp_hs}"
TRAIN_NPUS="${TRAIN_NPUS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-16}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://80.48.17.187:8001/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dspark_dsv4_flash_bestArch}"
LOG_DIR="$OUTPUT_DIR/logs"
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
TRAIN_CMD=(env -u LOCAL_RANK -u RANK -u WORLD_SIZE \
  ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
  --standalone --nproc_per_node "$NUM_TRAIN_NPUS" \
  "${TRAIN_ENTRY[@]}" \
  --verifier-name-or-path "$MODEL" \
  --data-path "$DATA_PATH" \
  --hidden-states-path "$HS_PATH" \
  --vllm-endpoint "$VLLM_ENDPOINT" \
  --save-path "$OUTPUT_DIR/checkpoints" \
  --target-hidden-state-format deepseek_v4_mean_hc_head \
  --draft-config examples/train/dsv4_flash_dense_config.json \
  --target-layer-ids 1 11 21 30 40 \
  --mask-token-id 128799 \
  --speculator-type dspark \
  --epochs 10 --lr 6e-5 --optimizer muon --scheduler-type linear \
  --logger tensorboard --log-dir "$LOG_DIR/tensorboard" --total-seq-len 3072 \
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
  --on-missing generate --on-generate delete)

# The smoke wrapper must wait for fresh to finish before starting resume.
if [[ "${TRAINING_SMOKE:-0}" == 1 ]]; then
  exec "${TRAIN_CMD[@]}"
fi
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_$$.log"
PID_FILE="$LOG_DIR/train.pid"
if [[ -f "$PID_FILE" ]]; then
  read -r previous_pid < "$PID_FILE" || true
  if [[ "${previous_pid:-}" =~ ^[1-9][0-9]*$ ]] && kill -0 "$previous_pid" 2>/dev/null; then
    echo "PID $previous_pid is still alive; check $PID_FILE before starting again." >&2
    exit 1
  fi
fi
nohup "${TRAIN_CMD[@]}" > "$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Training launched (PID $TRAIN_PID); check the log for startup errors."
echo "Log file: $LOG_FILE"
printf 'View log with: tail -f %q\n' "$LOG_FILE"
printf 'TensorBoard: tensorboard --logdir %q --host 127.0.0.1 --port 6006\n' "$LOG_DIR/tensorboard"
printf 'Stop with: kill -TERM %s\n' "$TRAIN_PID"
