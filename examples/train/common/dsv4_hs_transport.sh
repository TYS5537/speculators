#!/usr/bin/env bash
# Source from the DSV4 wrappers; does not launch services or touch shared storage.
configure_dsv4_hs_http_token() {
  # Preserve the supplied deployment token; environment overrides it on both ends.
  : "${DSV4_HS_HTTP_TOKEN:=8d4f1c7a9e2b6f30c5a1d8e74b9c2f61a7e5d3c8b0f2496e1c7a4d8b5f2e9031}"
  export DSV4_HS_HTTP_TOKEN
}

configure_dsv4_hs_transport() {
  local role="$1"
  HS_BACKEND="${HS_BACKEND:-file}"
  hs_transport_args=(--hidden-states-backend "$HS_BACKEND")
  case "$HS_BACKEND" in
    file) ;;
    http)
      if [[ "$role" == trainer ]]; then
        configure_dsv4_hs_http_token
        local endpoints=()
        read -r -a endpoints <<< "${HS_HTTP_ENDPOINTS:-http://${TARGET_MASTER_IP}:8002}"
        hs_transport_args+=(--hs-http-endpoints "${endpoints[@]}"
          --hs-http-timeout "${HS_TRANSFER_TIMEOUT:-180}"
          --hs-http-max-file-bytes "${HS_HTTP_MAX_FILE_BYTES:-536870912}")
      fi
      ;;
    mooncake)
      : "${MOONCAKE_NAMESPACE:?Set a fresh run ID (8-64 safe characters), identical on every target and trainer}"
      if [[ "$role" == server ]]; then
        export MOONCAKE_LOCAL_HOSTNAME="${MOONCAKE_LOCAL_HOSTNAME:-$TARGET_LOCAL_IP}"
        hs_transport_args+=(--mooncake-writer-threads "${MOONCAKE_WRITER_THREADS:-2}")
        if [[ -n "${MOONCAKE_PINNED_POOL_MIB:-}" ]]; then
          hs_transport_args+=(--mooncake-pinned-pool-mib "$MOONCAKE_PINNED_POOL_MIB")
        fi
        case "${MOONCAKE_COPY_EVENTS:-0}" in
          0) ;;
          1) hs_transport_args+=(--mooncake-copy-events) ;;
          *) echo 'MOONCAKE_COPY_EVENTS must be 0 or 1.' >&2; return 2 ;;
        esac
        MOONCAKE_GLOBAL_SEGMENT_GIB="${MOONCAKE_GLOBAL_SEGMENT_GIB:-4}"
        MOONCAKE_LOCAL_BUFFER_GIB="${MOONCAKE_LOCAL_BUFFER_GIB:-0.5}"
      else
        : "${MOONCAKE_LOCAL_HOSTNAME:?Set the reachable private IP of this trainer host}"
        export MOONCAKE_LOCAL_HOSTNAME
        hs_transport_args+=(--mooncake-target-dp-size "${TARGET_DP_SIZE:-2}")
        # Per DataLoader process, NOT a budget shared by all DDP ranks/workers.
        MOONCAKE_GLOBAL_SEGMENT_GIB="${MOONCAKE_GLOBAL_SEGMENT_GIB:-0.25}"
        MOONCAKE_LOCAL_BUFFER_GIB="${MOONCAKE_LOCAL_BUFFER_GIB:-0.25}"
      fi
      hs_transport_args+=(--mooncake-master "${MOONCAKE_MASTER:-${TARGET_MASTER_IP}:50051}"
        --mooncake-metadata-server "${MOONCAKE_METADATA_SERVER:-P2PHANDSHAKE}"
        --mooncake-protocol "${MOONCAKE_PROTOCOL:-tcp}"
        --mooncake-device-name "${MOONCAKE_DEVICE_NAME:-}"
        --mooncake-namespace "$MOONCAKE_NAMESPACE"
        --mooncake-global-segment-gib "$MOONCAKE_GLOBAL_SEGMENT_GIB"
        --mooncake-local-buffer-gib "$MOONCAKE_LOCAL_BUFFER_GIB"
        --hs-transfer-timeout "${HS_TRANSFER_TIMEOUT:-180}")
      ;;
    *) echo 'HS_BACKEND must be file, http or mooncake.' >&2; return 2 ;;
  esac
}
