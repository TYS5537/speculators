#!/usr/bin/env bash
# Optional CPU-only HS sidecar. Does not start, stop, or reconfigure vLLM.
# Bind a private interface explicitly for remote clients; HTTP requires a trusted LAN/VPN.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
: "${HS_PATH:=/home/s00969542/DSV4F/tmp_hs}"
# Saved deployment token matches the evaluator; override it on both hosts to rotate.
: "${DSV4_HS_HTTP_TOKEN:=8d4f1c7a9e2b6f30c5a1d8e74b9c2f61a7e5d3c8b0f2496e1c7a4d8b5f2e9031}"
export DSV4_HS_HTTP_TOKEN
exec python3 -m speculators_dsv4.hs_http_server \
  --hidden-states-path "$HS_PATH" \
  --host "${HS_HTTP_HOST:-80.48.17.187}" \
  --port "${HS_HTTP_PORT:-8002}" \
  --max-file-bytes "${HS_HTTP_MAX_FILE_BYTES:-536870912}"
