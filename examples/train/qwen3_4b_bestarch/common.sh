#!/usr/bin/env bash
# Source-only implementation shared by the small, named experiment entrypoints.
QWEN4B_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN4B_REPO="$(cd "$QWEN4B_DIR/../../.." && pwd)"
source "$QWEN4B_DIR/settings.sh"
source "$QWEN4B_DIR/../common/ascend_training_env.sh"
cd "$QWEN4B_REPO"

run_qwen4b_training() {
    local variant="$1" architecture="$2" policy="$3"
    local output_dir="${OUTPUT_DIR:-$OUTPUT_ROOT/$variant}"
    local lr="$LR" muon_lr="$MUON_LR"
    local weight_decay="$WEIGHT_DECAY" muon_weight_decay="$MUON_WEIGHT_DECAY"
    if [[ "$architecture" == dspark ]]; then
        lr="$DSPARK_LR"
        muon_lr="$DSPARK_LR"
        weight_decay="$DSPARK_WEIGHT_DECAY"
        muon_weight_decay="$DSPARK_WEIGHT_DECAY"
    fi
    local -a train_devices
    IFS=',' read -r -a train_devices <<< "$TRAIN_NPUS"
    local num_train_npus="${#train_devices[@]}"
    if [[ ! "$TRAIN_NPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "TRAIN_NPUS must be comma-separated device IDs" >&2
        return 2
    fi
    configure_ascend_training_env
    local -a train_args=(
        --training-recipe legacy
        --loss-implementation legacy
        --optimizer muon
        --muon-parameter-policy "$policy"
        --lr "$lr" --muon-lr "$muon_lr"
        --weight-decay "$weight_decay" --muon-weight-decay "$muon_weight_decay"
        --verifier-name-or-path "$MODEL"
        --data-path "$DATA_PATH"
        --vllm-endpoint "$VLLM_ENDPOINT"
        --save-path "$output_dir/checkpoints"
        --log-dir "$output_dir/tensorboard" --run-name "$variant"
        --no-resume-from-checkpoint
        --seed "$SEED" --epochs "$EPOCHS" --logger "$LOGGER"
        --total-seq-len "$SEQ_LENGTH" --speculator-type "$architecture"
        --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS"
        --num-layers "$NUM_LAYERS" --draft-attn-impl "$DRAFT_ATTN_IMPL"
        --target-layer-ids "${TARGET_LAYER_IDS[@]}"
        --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE"
        --enable-confidence-head --confidence-head-with-markov
        --loss-fn "$LOSS_FN"
        --confidence-head-alpha 1.0 --confidence-length-alpha 0.0
        --confidence-loss-weighting match-draft --no-confidence-detach-features
        --first-error-focal-alpha 0.0 --adaptive-loss none
        --no-ssal-curriculum --ssal-curriculum-start 0.1 --ssal-curriculum-end 0.6
        --on-missing generate --on-generate delete
    )
    if [[ -n "$DRAFT_VOCAB_SIZE" ]]; then
        train_args+=(--draft-vocab-size "$DRAFT_VOCAB_SIZE")
    fi
    if [[ "$architecture" == mmuse ]]; then
        train_args+=(
            --full-attention-indices "${MMUSE_FULL_ATTENTION_INDICES[@]}"
            "${MMUSE_BESTARCH_ARGS[@]}"
        )
    else
        train_args+=(
            --scheduler-type "$DSPARK_SCHEDULER_TYPE"
            --scheduler-warmup-ratio "$DSPARK_WARMUP_RATIO"
            --full-attention-indices "${DSPARK_FULL_ATTENTION_INDICES[@]}"
        )
    fi
    local -a command=(env "ASCEND_RT_VISIBLE_DEVICES=$TRAIN_NPUS"
        torchrun --standalone --nproc_per_node "$num_train_npus"
        -m speculators.train "${train_args[@]}")
    if [[ "${DUMP_CONFIG:-0}" == 1 ]]; then
        python -m speculators.train "${train_args[@]}" --dump-config
        return
    fi
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    if [[ ! -d "$DATA_PATH" ]]; then
        echo "Prepared Arrow data not found: $DATA_PATH (see settings.sh)" >&2
        return 2
    fi
    # These are fresh-run comparisons, never an implicit cross-policy resume.
    if [[ -e "$output_dir/checkpoints" || -e "$output_dir/logs/train.pid" ]]; then
        echo "Run directory already used: $output_dir; choose a new OUTPUT_DIR" >&2
        return 2
    fi
    if ! curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf \
        "${VLLM_ENDPOINT%/}/models" > /dev/null; then
        echo "Target unavailable: start server.sh first ($VLLM_ENDPOINT)" >&2
        return 2
    fi
    mkdir -p "$output_dir/logs"
    local log_file="$output_dir/logs/train_$(date +%Y%m%d_%H%M%S).log"
    nohup "${command[@]}" > "$log_file" 2>&1 &
    local train_pid=$!
    printf '%s\n' "$train_pid" > "$output_dir/logs/train.pid"
    echo "Started $variant, PID $train_pid, NPUs $TRAIN_NPUS"
    echo "Log: $log_file"
    echo "TensorBoard: $output_dir/tensorboard"
    echo "Stop: kill $train_pid"
    echo "Run the three experiments sequentially; they share the training NPUs."
}
