#!/usr/bin/env bash
# Optional CPU-only HS sidecar. Does not start, stop, or reconfigure vLLM.
# Bind a private interface explicitly for remote clients; HTTP requires a trusted LAN/VPN.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:${PYTHONPATH:-}"
HS_PATH="${HS_PATH:-/home/s00969542/DSV4F/tmp_hs}"
HS_HTTP_TRAINING="${HS_HTTP_TRAINING:-1}"
source "$REPO_ROOT/examples/train/common/dsv4_hs_transport.sh"
configure_dsv4_hs_http_token
sidecar_args=()
case "$HS_HTTP_TRAINING" in
  0) ;;
  1)
    sidecar_args+=(--training)
    ;;
  *) echo 'HS_HTTP_TRAINING must be 0 or 1.' >&2; exit 2 ;;
esac
exec python3 -m speculators_dsv4.hs_http_server \
  "${sidecar_args[@]}" \
  --hidden-states-path "$HS_PATH" \
  --host "${HS_HTTP_HOST:-80.48.17.187}" \
  --port "${HS_HTTP_PORT:-8002}" \
  --max-file-bytes "${HS_HTTP_MAX_FILE_BYTES:-536870912}"
