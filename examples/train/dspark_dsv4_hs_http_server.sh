#!/usr/bin/env bash
# Optional CPU-only HS sidecar. Does not start, stop, or reconfigure vLLM.
# Bind a private interface explicitly for remote clients; HTTP requires a trusted LAN/VPN.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
: "${HS_PATH:?Set HS_PATH to the existing target HS directory}"
: "${DSV4_HS_HTTP_TOKEN:?Export a random bearer token of at least 32 characters}"
export DSV4_HS_HTTP_TOKEN
exec python3 -m speculators_dsv4.hs_http_server \
  --hidden-states-path "$HS_PATH" \
  --host "${HS_HTTP_HOST:-127.0.0.1}" \
  --port "${HS_HTTP_PORT:-8002}" \
  --max-file-bytes "${HS_HTTP_MAX_FILE_BYTES:-536870912}"
