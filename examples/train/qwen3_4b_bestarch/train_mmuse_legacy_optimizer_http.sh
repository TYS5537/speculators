#!/bin/bash
# MMuse Qwen3-4B, block 15: .135 and .137, each with 2 target + 6 training NPUs.
# Run from the repository root. Edit this file directly; no shared settings.
set -euo pipefail

export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# Level 2 can raise NPU workspace memory peaks during large-block backward.
export TASK_QUEUE_ENABLE=1 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0

# ============ Configuration ============
MODEL="../../Qwen3-4B"
DATA_PATH="../../datasets/open_perfectblend_qwen3_4b_700k"
OUTPUT_DIR="./output/qwen3_4b_bestarch/mmuse_legacy_http_2x6_135_137_bs15"
RUN_NAME="mmuse_legacy_http_2x6_135_137_bs15"
TARGET_HOST="127.0.0.1"  # Each host uses its own local server_http.sh instance.
VLLM_PORT=8001
HS_HTTP_PORT=8002
HTTP_TOKEN="ff7b93b9f2e5be1e3eb1c5448b37bb15d158b7b31c1c64a2ae2e6dd5244f9083"  # Shared with the server; keep the same value on all hosts.
VLLM_ENDPOINT="http://${TARGET_HOST}:${VLLM_PORT}/v1"
HS_HTTP_ENDPOINTS=("http://80.5.5.135:${HS_HTTP_PORT}" "http://80.5.5.137:${HS_HTTP_PORT}")
HS_HTTP_TIMEOUT=120
REQUEST_TIMEOUT=600
# The shared HTTP_TOKEN is exported to child processes during launch.
export NO_PROXY="localhost,127.0.0.1,${TARGET_HOST}"
export no_proxy="$NO_PROXY"
TRAIN_NPUS="2,3,4,5,6,7"
NUM_TRAIN_NPUS=6  # LOCAL device count on each machine, not the global total.
NUM_TRAIN_NODES=2
TRAIN_NODE_IPS=("80.5.5.135" "80.5.5.137")  # Ordered: ranks 0, 1.
TRAIN_NODE_RANK="auto"  # Resolve from a locally assigned address in TRAIN_NODE_IPS.
TRAIN_MASTER_ADDR="80.5.5.135"
TRAIN_MASTER_PORT=34655
TRAIN_IFNAME="auto"  # Resolve the interface owning that exact local IP.
TRAIN_LOCAL_IP="auto"  # Resolve to this host's .135 or .137 address.

SEQ_LENGTH=3072
EPOCHS=10
SEED=42
LOGGER="tensorboard"
LR=6e-5
WEIGHT_DECAY=0.01
MUON_LR=6e-4
MUON_WEIGHT_DECAY=0.1
MUON_PARAMETER_POLICY="legacy"
BLOCK_SIZE=15
MAX_ANCHORS=512
NUM_LAYERS=5
TARGET_LAYER_IDS=(1 9 17 25 33)
FULL_ATTENTION_INDICES=(0 1 2 3 4)
DRAFT_ATTN_IMPL="sdpa"
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'

# Bestarch enhancements; switches are written explicitly in the command below.
CORRECTION_HIDDEN_SIZE=768
CORRECTION_RANK=256
CORRECTION_NUM_LAYERS=1
CORRECTION_NUM_HEADS=8
CORRECTION_GATE_BIAS=0
CORRECTION_HIDDEN_AUX_WEIGHT=0.1
CORRECTION_MARKOV_GATE_BIAS=-2.0
CONV_KERNEL_SIZE=2
CONV_GROUP_SIZE=16
SELECTOR_RANK=256
SELECTOR_TOP_K=16
SELECTOR_LOSS_WEIGHT=0.1

# ============ Launch ============
if [[ "$TARGET_HOST" == "CHANGE_ME" || -z "$TARGET_HOST" ]]; then
    echo "Set TARGET_HOST to the target machine's private IP/DNS." >&2
    exit 2
fi
if [[ ${#HTTP_TOKEN} -lt 32 ]]; then
    echo "Set the same HTTP_TOKEN in both HTTP scripts (at least 32 characters)." >&2
    exit 2
fi
export SPECULATORS_HS_HTTP_TOKEN="$HTTP_TOKEN"
if [[ ! "$NUM_TRAIN_NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_TRAIN_NODES must be positive." >&2
    exit 2
fi
if [[ "$TRAIN_NODE_RANK" == "auto" || "$TRAIN_LOCAL_IP" == "auto" || "$TRAIN_IFNAME" == "auto" ]]; then
    if (( ${#TRAIN_NODE_IPS[@]} != NUM_TRAIN_NODES )); then
        echo "TRAIN_NODE_IPS must contain one ordered IP per training node." >&2
        exit 2
    fi
    if ! command -v ip >/dev/null 2>&1 || ! HOST_IPV4="$(ip -o -4 addr show)"; then
        echo "Cannot read local IPv4 interfaces; install iproute2 or set all TRAIN_NODE_RANK/TRAIN_LOCAL_IP/TRAIN_IFNAME explicitly." >&2
        exit 2
    fi
    LOCAL_MATCHES=()
    while read -r ADDR_INDEX ADDR_IFNAME ADDR_FAMILY ADDR_PREFIX ADDR_DETAILS; do
        [[ "$ADDR_FAMILY" == "inet" ]] || continue
        for NODE_INDEX in "${!TRAIN_NODE_IPS[@]}"; do
            if [[ "${ADDR_PREFIX%/*}" == "${TRAIN_NODE_IPS[$NODE_INDEX]}" ]]; then
                LOCAL_MATCHES+=("$NODE_INDEX|${ADDR_PREFIX%/*}|${ADDR_IFNAME%%@*}")
            fi
        done
    done <<< "$HOST_IPV4"
    if (( ${#LOCAL_MATCHES[@]} != 1 )); then
        echo "Expected exactly one local TRAIN_NODE_IPS address; found ${#LOCAL_MATCHES[@]}. Check the host/container network." >&2
        exit 2
    fi
    IFS='|' read -r DETECTED_RANK DETECTED_IP DETECTED_IFNAME <<< "${LOCAL_MATCHES[0]}"
    if [[ "$TRAIN_NODE_RANK" != "auto" && "$TRAIN_NODE_RANK" != "$DETECTED_RANK" ]] ||
       [[ "$TRAIN_LOCAL_IP" != "auto" && "$TRAIN_LOCAL_IP" != "$DETECTED_IP" ]] ||
       [[ "$TRAIN_IFNAME" != "auto" && "$TRAIN_IFNAME" != "$DETECTED_IFNAME" ]]; then
        echo "TRAIN_NODE_RANK/TRAIN_LOCAL_IP/TRAIN_IFNAME conflict with the detected local node ($DETECTED_RANK, $DETECTED_IP, $DETECTED_IFNAME)." >&2
        exit 2
    fi
    TRAIN_NODE_RANK="$DETECTED_RANK"
    TRAIN_LOCAL_IP="$DETECTED_IP"
    TRAIN_IFNAME="$DETECTED_IFNAME"
fi
if [[ ! "$TRAIN_NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] ||
   (( TRAIN_NODE_RANK >= NUM_TRAIN_NODES )); then
    echo "TRAIN_NODE_RANK must be in [0, NUM_TRAIN_NODES)." >&2
    exit 2
fi
if [[ ! "$TRAIN_MASTER_PORT" =~ ^[1-9][0-9]*$ ]] ||
   (( TRAIN_MASTER_PORT > 65535 )); then
    echo "TRAIN_MASTER_PORT must be in [1, 65535]." >&2
    exit 2
fi
if (( NUM_TRAIN_NODES > 1 )); then
    for HOST_IP in "$TRAIN_MASTER_ADDR" "$TRAIN_LOCAL_IP"; do
        if [[ -z "$HOST_IP" || "$HOST_IP" == "CHANGE_ME" ||
              "$HOST_IP" == "localhost" || "$HOST_IP" == 127.* ||
              "$HOST_IP" == "::1" || "$HOST_IP" == "0.0.0.0" ]]; then
            echo "Multi-node training requires reachable TRAIN_MASTER_ADDR and TRAIN_LOCAL_IP." >&2
            exit 2
        fi
    done
    if [[ -z "$TRAIN_IFNAME" || "$TRAIN_IFNAME" == "lo" || "$TRAIN_IFNAME" == "CHANGE_ME" ]]; then
        echo "Set TRAIN_IFNAME to this machine's Host communication NIC." >&2
        exit 2
    fi
    if (( ${#HS_HTTP_ENDPOINTS[@]} < NUM_TRAIN_NODES )); then
        echo "List every machine's HS service in HS_HTTP_ENDPOINTS to verify all target copies." >&2
        exit 2
    fi
    export HCCL_SOCKET_IFNAME="=$TRAIN_IFNAME"
    export HCCL_IF_IP="$TRAIN_LOCAL_IP"
fi
export GLOO_SOCKET_IFNAME="$TRAIN_IFNAME"
echo "Training node $TRAIN_NODE_RANK: $TRAIN_LOCAL_IP on $TRAIN_IFNAME; master $TRAIN_MASTER_ADDR:$TRAIN_MASTER_PORT"
# Fresh-run comparisons: never resume a different optimizer's state.
if [[ ! "$TRAIN_NPUS" =~ ^[0-9]+(,[0-9]+)*$ ||
      ! "$NUM_TRAIN_NPUS" =~ ^[1-9][0-9]*$ ||
      "$NUM_TRAIN_NPUS" != "$(awk -F, '{print NF}' <<< "$TRAIN_NPUS")" ]]; then
    echo "TRAIN_NPUS and NUM_TRAIN_NPUS must list/count the same devices." >&2
    exit 2
fi
if [[ ! -d "$DATA_PATH" ]]; then
    echo "Prepared Arrow data not found: $DATA_PATH" >&2
    exit 2
fi
if [[ -e "$OUTPUT_DIR/checkpoints" || -e "$OUTPUT_DIR/logs/train.pid" ||
      -e "$OUTPUT_DIR/logs/node${TRAIN_NODE_RANK}/train.pid" ]]; then
    echo "Run directory already used: $OUTPUT_DIR; choose a new OUTPUT_DIR." >&2
    exit 2
fi
if ! curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf \
    "${VLLM_ENDPOINT%/}/models" > /dev/null; then
    echo "Target unavailable: start server_http.sh on the target machine ($VLLM_ENDPOINT)." >&2
    exit 2
fi

# Verify every target's checkpoint/layer order before joining the training job.
python -m speculators.remote_http check \
    --model "$MODEL" --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --hidden-states-path "$OUTPUT_DIR/hs_downloads" \
    --hs-http-endpoints "${HS_HTTP_ENDPOINTS[@]}" --timeout "$HS_HTTP_TIMEOUT"

LOG_DIR="$OUTPUT_DIR/logs/node${TRAIN_NODE_RANK}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train.pid"
# Keep each host's launch topology alongside logs; never record the HTTP token.
printf '%s\n' \
    "node=$TRAIN_NODE_RANK/$NUM_TRAIN_NODES local_npus=$TRAIN_NPUS local_workers=$NUM_TRAIN_NPUS" \
    "master=$TRAIN_MASTER_ADDR:$TRAIN_MASTER_PORT interface=$TRAIN_IFNAME local_ip=$TRAIN_LOCAL_IP" \
    "vllm=$VLLM_ENDPOINT" "hs_http=${HS_HTTP_ENDPOINTS[*]}" \
    > "$LOG_DIR/topology.txt"

nohup env ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun \
    --nnodes "$NUM_TRAIN_NODES" --node_rank "$TRAIN_NODE_RANK" \
    --master_addr "$TRAIN_MASTER_ADDR" --master_port "$TRAIN_MASTER_PORT" \
    --nproc_per_node "$NUM_TRAIN_NPUS" --rdzv_id "$RUN_NAME" --max_restarts 0 \
    -m speculators.train \
    --training-recipe legacy \
    --loss-implementation legacy \
    --optimizer muon \
    --muon-parameter-policy "$MUON_PARAMETER_POLICY" \
    --muon-lr "$MUON_LR" --muon-weight-decay "$MUON_WEIGHT_DECAY" \
    --lr "$LR" --weight-decay "$WEIGHT_DECAY" \
    --verifier-name-or-path "$MODEL" \
    --data-path "$DATA_PATH" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --hidden-states-backend http --hidden-states-path "$OUTPUT_DIR/hs_downloads" \
    --hs-http-endpoints "${HS_HTTP_ENDPOINTS[@]}" --hs-http-timeout "$HS_HTTP_TIMEOUT" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --fsdp-shard --activation-checkpointing \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --log-dir "$OUTPUT_DIR/tensorboard" --run-name "$RUN_NAME" \
    --no-resume-from-checkpoint \
    --seed "$SEED" --epochs "$EPOCHS" --logger "$LOGGER" \
    --total-seq-len "$SEQ_LENGTH" --speculator-type mmuse \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" --draft-attn-impl "$DRAFT_ATTN_IMPL" \
    --target-layer-ids "${TARGET_LAYER_IDS[@]}" \
    --full-attention-indices "${FULL_ATTENTION_INDICES[@]}" \
    --markov-rank "$MARKOV_RANK" --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-correction-head \
    --correction-output-mode logits \
    --correction-hidden-size "$CORRECTION_HIDDEN_SIZE" \
    --correction-rank "$CORRECTION_RANK" \
    --correction-lm-head-fusion \
    --correction-num-layers "$CORRECTION_NUM_LAYERS" \
    --correction-num-heads "$CORRECTION_NUM_HEADS" \
    --correction-gate-bias "$CORRECTION_GATE_BIAS" \
    --correction-hidden-aux-loss \
    --correction-hidden-aux-weight "$CORRECTION_HIDDEN_AUX_WEIGHT" \
    --correction-hidden-feedback \
    --selector-correction-feedback corrected \
    --correction-project-corrected-hidden \
    --correction-with-markov \
    --correction-markov-gate-bias "$CORRECTION_MARKOV_GATE_BIAS" \
    --no-correction-rollout-metrics \
    --no-correction-base-diagnostics \
    --dflash-context-residual \
    --dflash-block-position-embedding \
    --dflash-gated-layer-fusion \
    --dflash2-dynamic-conv \
    --dflash2-conv-kernel-size "$CONV_KERNEL_SIZE" \
    --dflash2-conv-group-size "$CONV_GROUP_SIZE" \
    --dflash2-candidate-selector \
    --dflash2-selector-rank "$SELECTOR_RANK" \
    --dflash2-selector-top-k "$SELECTOR_TOP_K" \
    --dflash2-selector-greedy \
    --dflash2-selector-loss-weight "$SELECTOR_LOSS_WEIGHT" \
    --enable-confidence-head --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha 1.0 --confidence-length-alpha 0.0 \
    --confidence-loss-weighting match-draft --no-confidence-detach-features \
    --first-error-focal-alpha 0.0 --adaptive-loss none \
    --no-ssal-curriculum --ssal-curriculum-start 0.1 --ssal-curriculum-end 0.6 \
    --on-missing generate --on-generate delete \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"
echo "Started $RUN_NAME, node $TRAIN_NODE_RANK/$NUM_TRAIN_NODES, PID $TRAIN_PID, NPUs $TRAIN_NPUS"
echo "Global training workers: $((NUM_TRAIN_NODES * NUM_TRAIN_NPUS)); all nodes must start this same job."
echo "Log: $LOG_FILE"
echo "TensorBoard: $OUTPUT_DIR/tensorboard"
echo "Stop: kill $TRAIN_PID"
echo "Target HTTP: $VLLM_ENDPOINT; hidden states: ${HS_HTTP_ENDPOINTS[*]}"
echo "The remote target stays running independently of this trainer."
