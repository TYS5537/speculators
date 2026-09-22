#!/usr/bin/env bash
# Shared baseline DSpark argv for the two full online Qwen examples.
# Reads their configuration variables and replaces DSPARK_TRAIN_ARGS on each call.
# Optional arguments (e.g. attention implementation) go before target layer IDs.
# Device selection, data preparation and server/trainer lifecycles stay in callers.
build_dspark_online_train_args() {
    DSPARK_TRAIN_ARGS=(
        --verifier-name-or-path "$MODEL"
        --data-path "$OUTPUT_DIR"
        --vllm-endpoint "http://localhost:${VLLM_PORT}/v1"
        --save-path "$OUTPUT_DIR/checkpoints"
        --draft-vocab-size "$DRAFT_VOCAB_SIZE"
        --epochs "$EPOCHS"
        --lr "$LR"
        --total-seq-len "$SEQ_LENGTH"
        --speculator-type "$SPECULATOR_TYPE"
        --block-size "$BLOCK_SIZE"
        --max-anchors "$MAX_ANCHORS"
        --num-layers "$NUM_LAYERS"
        "$@"
        # Preserve the examples' space-separated layer list expansion.
        --target-layer-ids $TARGET_LAYER_IDS
        --markov-rank "$MARKOV_RANK"
        --markov-head-type "$MARKOV_HEAD_TYPE"
        --enable-confidence-head
        --confidence-head-with-markov
        --loss-fn "$LOSS_FN"
        --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA"
        --confidence-loss-weighting "$CONFIDENCE_LOSS_WEIGHTING"
        --on-missing generate
        --on-generate delete
    )
}
