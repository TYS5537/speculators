"""Versioned DSV4 block protocol: stateless v1/v2, immutable KV snapshots v3."""

BLOCK_PROTOCOL_VERSION = 1
# Version 1 remains the full-probability compatibility path. Version 2 explicitly
# requests compact greedy results and/or synchronized diagnostic timings.
BLOCK_EXTENDED_VERSION = 2
BLOCK_KV_VERSION = 3
KV_MAX_ENTRIES = 64
KV_CONNECTOR = "DSV4CachedBlockVerifyConnector"
KV_CONNECTOR_MODULE = "speculators_dsv4.cached_connector"
BLOCK_PROFILE_STAGES = ("server_forward", "server_head", "server_packet_prepare")
BLOCK_REQUEST_KEY = "dsv4_block_verify"
BLOCK_CONNECTOR = "DSV4BlockVerifyConnector"
BLOCK_CONNECTOR_MODULE = "speculators_dsv4.block_connector"
GREEDY_REQUEST_KEY = "dsv4_greedy_trace"
GREEDY_VERSION = 1
REPLAY_CONNECTOR = "DSV4ReplayConnector"
REPLAY_CONNECTOR_MODULE = "speculators_dsv4.replay_connector"


def validate_block_request(value, prompt_length):
    """Return validated zero-based logits/HS offsets, rejecting silent fallback."""
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("Block verification requires a nonempty token prefix")
    if not isinstance(value, dict):
        raise ValueError("Missing or invalid DSV4 block verification request")
    version = value.get("version")
    if type(version) is not int or version not in (
        BLOCK_PROTOCOL_VERSION,
        BLOCK_EXTENDED_VERSION,
        BLOCK_KV_VERSION,
    ):
        raise ValueError("Unsupported DSV4 block verification protocol")
    fields = {
        "version",
        "logits_start",
        "hidden_start",
    }
    fields.update(
        {
            BLOCK_PROTOCOL_VERSION: (),
            BLOCK_EXTENDED_VERSION: ("output_mode", "profile"),
            BLOCK_KV_VERSION: ("output_mode", "profile", "cache"),
        }[version]
    )
    if set(value) != fields:
        raise ValueError("Missing or invalid DSV4 block verification request")
    if version >= BLOCK_EXTENDED_VERSION and (
        value["output_mode"] not in ("logprobs", "greedy")
        or type(value["profile"]) is not bool
    ):
        raise ValueError("Invalid block output_mode or profile setting")
    logits_start = value["logits_start"]
    hidden_start = value["hidden_start"]
    if type(logits_start) is not int or not 0 <= logits_start < prompt_length:
        raise ValueError("Block logits_start must select a nonempty prompt suffix")
    if type(hidden_start) is not int or not 0 <= hidden_start <= prompt_length:
        raise ValueError("Block hidden_start is outside the token prefix")
    if version == BLOCK_KV_VERSION:
        validate_cache_options(value["cache"])
    return logits_start, hidden_start


def validate_cache_options(value):
    """Opaque immutable snapshot capabilities, never filesystem paths."""
    import re  # noqa: PLC0415

    def valid_key(key):
        return isinstance(key, str) and re.fullmatch(r"[0-9a-f]{32}", key) is not None

    if (
        not isinstance(value, dict)
        or set(value) != {"read", "write", "release"}
        or (value["read"] is not None and not valid_key(value["read"]))
        or not valid_key(value["write"])
        or not isinstance(value["release"], list)
        or len(value["release"]) > KV_MAX_ENTRIES
        or any(not valid_key(key) for key in value["release"])
        or len(set(value["release"])) != len(value["release"])
        or value["write"] == value["read"]
        or value["write"] in value["release"]
    ):
        raise ValueError("Invalid DSV4 KV snapshot capabilities")
