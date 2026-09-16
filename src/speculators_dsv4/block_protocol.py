"""Small, versioned contract for stateless DSV4 block verification."""

BLOCK_PROTOCOL_VERSION = 1
BLOCK_REQUEST_KEY = "dsv4_block_verify"
BLOCK_CONNECTOR = "DSV4BlockVerifyConnector"
BLOCK_CONNECTOR_MODULE = "speculators_dsv4.block_connector"


def validate_block_request(value, prompt_length):
    """Return validated zero-based logits/HS offsets, rejecting silent fallback."""
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError("Block verification requires a nonempty token prefix")
    if not isinstance(value, dict) or set(value) != {
        "version",
        "logits_start",
        "hidden_start",
    }:
        raise ValueError("Missing or invalid DSV4 block verification request")
    if type(value["version"]) is not int or value["version"] != BLOCK_PROTOCOL_VERSION:
        raise ValueError("Unsupported DSV4 block verification protocol")
    logits_start = value["logits_start"]
    hidden_start = value["hidden_start"]
    if type(logits_start) is not int or not 0 <= logits_start < prompt_length:
        raise ValueError("Block logits_start must select a nonempty prompt suffix")
    if type(hidden_start) is not int or not 0 <= hidden_start <= prompt_length:
        raise ValueError("Block hidden_start is outside the token prefix")
    return logits_start, hidden_start
