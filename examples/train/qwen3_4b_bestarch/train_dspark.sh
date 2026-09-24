#!/usr/bin/env bash
# DSpark: old recipe, changing only peak LR, weight decay, schedule and attention.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_qwen4b_training dspark_custom dspark legacy
