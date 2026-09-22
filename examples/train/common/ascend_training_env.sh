#!/usr/bin/env bash
# Sourcing only defines the preset; callers opt in at their original setup point.
# For training, not standalone vLLM servers with their own task-queue settings.
configure_ascend_training_env() {
    export OMP_PROC_BIND=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VE_OMP_NUM_THREADS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=2 ACLNN_CACHE_LIMIT=100000 NPU_ASD_ENABLE=0 ASCEND_LAUNCH_BLOCKING=0
}
