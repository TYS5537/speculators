#!/usr/bin/env bash
# Shared settings for the three Qwen3-4B comparisons and their target server.
# Relative model/data/output paths are resolved from the repository root.
MODEL="${MODEL:-../../Qwen3-4B}"
DATA_PATH="${DATA_PATH:-../../datasets/open_perfectblend_qwen3_4b_700k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/qwen3_4b_bestarch}"
TARGET_LAYER_IDS=(1 9 17 25 33)
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:${VLLM_PORT}/v1}"
VLLM_NPUS="${VLLM_NPUS:-0,1}"
VLLM_EXTRA_ARGS=(--data-parallel-size 2)
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}"

# DSpark-only changes, expressed entirely through the existing training CLI.
# Optimizer type/grouping, data, packed batch and losses stay on the old recipe.
# Both optimizer groups use this peak LR and weight decay; no implicit 10x LR.
DSPARK_LR="${DSPARK_LR:-6e-4}"
DSPARK_WEIGHT_DECAY="${DSPARK_WEIGHT_DECAY:-0.0}"
DSPARK_SCHEDULER_TYPE="${DSPARK_SCHEDULER_TYPE:-cosine}"
DSPARK_WARMUP_RATIO="${DSPARK_WARMUP_RATIO:-0.04}"
DSPARK_FULL_ATTENTION_INDICES=(0 1 2 3 4)

# Shared original training budget. LR/decay below apply to the two MMuse runs;
# DSpark overrides only the four settings above. Both keep loss gamma=7 via legacy.
SEQ_LENGTH="${SEQ_LENGTH:-3072}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-6e-5}"
# Pin the old effective Muon LR in BOTH MMuse runs. Only grouping changes.
MUON_LR="${MUON_LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-0.1}"
SEED="${SEED:-42}"
LOGGER="${LOGGER:-tensorboard}"
BLOCK_SIZE=7
MAX_ANCHORS=512
NUM_LAYERS=5
DRAFT_ATTN_IMPL=sdpa
MARKOV_RANK=256
MARKOV_HEAD_TYPE=vanilla
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
# The supplied bestarch script did NOT force a 32k draft vocabulary. Leave empty
# to reuse existing d2t.npy/t2d.npy, or use the full target vocab if absent.
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-}"

# Exact enabled extensions from the supplied bestarch script, now under MMuse.
# No Correction MoE, DFly fusion or KV-projection experiment is enabled here.
# Both MMuse comparisons now use full attention in all five draft layers too.
MMUSE_FULL_ATTENTION_INDICES=(0 1 2 3 4)
MMUSE_BESTARCH_ARGS=(
    --enable-correction-head
    --correction-output-mode logits
    --correction-hidden-size 768
    --correction-rank 256
    --correction-lm-head-fusion
    --correction-num-layers 1
    --correction-num-heads 8
    --correction-gate-bias 0
    --correction-hidden-aux-loss
    --correction-hidden-aux-weight 0.1
    --correction-hidden-feedback
    --selector-correction-feedback corrected
    --correction-project-corrected-hidden
    --correction-with-markov
    --correction-markov-gate-bias -2.0
    --no-correction-rollout-metrics
    --no-correction-base-diagnostics
    --dflash-context-residual
    --dflash-block-position-embedding
    --dflash-gated-layer-fusion
    --dflash2-dynamic-conv
    --dflash2-conv-kernel-size 2
    --dflash2-conv-group-size 16
    --dflash2-candidate-selector
    --dflash2-selector-rank 256
    --dflash2-selector-top-k 16
    --dflash2-selector-greedy
    --dflash2-selector-loss-weight 0.1
)

# Preserve the supplied local networking environment; users may override it.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-34655}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54}"
export no_proxy="$NO_PROXY"
