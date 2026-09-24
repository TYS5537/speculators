#!/usr/bin/env bash
# Supplied bestarch, retaining the old Muon grouping (including Markov factors).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
run_qwen4b_training mmuse_legacy_optimizer mmuse legacy
