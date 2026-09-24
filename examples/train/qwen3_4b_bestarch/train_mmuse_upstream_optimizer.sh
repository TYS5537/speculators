#!/usr/bin/env bash
# Same bestarch/init/loss/LRs; only route Markov factors to upstream's AdamW group.
# This does NOT enable the complete upstream training recipe or PR #1066.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_qwen4b_training mmuse_upstream_optimizer mmuse upstream
