"""Shared response-regeneration helpers; entry points keep their row policies."""

import hashlib
import json
import re
from typing import Any

import aiohttp

from speculators.data_generation.vllm_client import InvalidResponseError, with_retries

_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
SERVER_ERROR_STATUS = 500
RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}


def _message_role_content(m: dict) -> tuple[str | None, Any]:
    """Canonical ``(role, content)`` for a message across the role/content and
    from/value schemas. ``role`` collapses ``human`` to ``user``; ``system`` and
    ``tool`` pass through; anything else (assistant/gpt) returns ``None``."""
    role = m.get("role") or m.get("from")
    content = m.get("content")
    if content is None:
        content = m.get("value")
    if role in ("user", "human"):
        return "user", content
    if role in ("system", "tool"):
        return role, content
    return None, content


def _maybe_json(value: Any):
    """Best-effort JSON decode; return None if the value is not valid JSON."""
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _list_field(row, key) -> list | None:
    """Non-empty list from ``row[key]`` (JSON-decoded if a string), else None."""
    value = row.get(key)
    if isinstance(value, str):
        value = _maybe_json(value)
    return value if isinstance(value, list) and value else None


def extract_tools(row) -> list | None:
    """Return the OpenAI-style ``tools`` schema for a row, or ``None``.

    Reads the ``tools`` column -- a list, or a JSON-string encoding one, as the
    Hermes function-calling dataset stores it. A row that declares a ``tools``
    field we cannot read as a list raises ``ValueError`` rather than silently
    regenerating tool-free; a tool-free row returns ``None``.
    """
    tools = _list_field(row, "tools")
    if tools:
        return tools
    if row.get("tools") not in (None, "", [], {}):
        raise ValueError("a tools field is present but not a usable list")
    return None


def _tool_result_names(content: Any) -> list[str]:
    """Tool names a cached result answers, for the splice name-match guard.

    Hermes embeds the answering tool's name in each ``<tool_response>`` payload.
    Returns ``[]`` when no name can be read, which disables the guard for that
    result rather than blocking the splice on our own parse miss.
    """
    if not isinstance(content, str):
        return []
    names = []
    for block in _TOOL_RESPONSE_RE.findall(content):
        obj = _maybe_json(block)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            names.append(obj["name"])
    return names


def _is_present(value: Any) -> bool:
    """Return True for a usable identifier (not None / not empty string)."""
    return value not in (None, "")


def _content_hash(row: dict[str, Any]) -> str:
    """Deterministic hash of a row, used when it has no explicit id."""
    payload = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    return "hash_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _primary_identifier(row: dict[str, Any]) -> str:
    """Return a stable primary id for a dataset row.

    Prefers an explicit ``id``/``uuid``; otherwise a deterministic content hash.
    Unlike a streaming enumeration index, this key does not shift when
    ``--limit``/``--language-filter`` or the input order change, so ``--resume``
    stays correct across runs.
    """
    for field in ("id", "uuid"):
        value = row.get(field)
        if _is_present(value):
            return str(value)
    return _content_hash(row)


@with_retries
async def _post_chat(
    session: aiohttp.ClientSession,
    endpoint: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """POST one chat-completion request and return the parsed response.

    Wrapped by ``with_retries`` (adds a ``max_retries`` kwarg): transient
    failures — network errors and transient HTTP statuses (408/409/425/429/5xx)
    — are retried with exponential backoff. Permanent non-2xx replies (e.g.
    400/404) raise ``InvalidResponseError``, which ``with_retries`` never
    retries, so they fail fast. A non-2xx reply is surfaced with its status and
    a short body so the caller does not record a bare ``KeyError('choices')``.
    """
    async with session.post(endpoint, json=payload) as response:
        if not response.ok:
            body = (await response.text())[:500]
            message = f"HTTP {response.status} from {endpoint}: {body}"
            # Retry transient statuses (408/409/425/429/5xx); fail fast otherwise.
            if (
                response.status >= SERVER_ERROR_STATUS
                or response.status in RETRYABLE_HTTP_STATUSES
            ):
                raise RuntimeError(message)
            raise InvalidResponseError(message)
        return await response.json()


def build_boundary_sample(
    prompt_token_ids: list[int],
    completion_token_ids: list[int],
) -> tuple[list[int], list[int]]:
    """Build one training sample: prompt (loss_mask 0) + generated tokens (1).

    The generation boundary is the mask -- no ``{% generation %}`` markers, no regex.
    """
    input_ids = [*prompt_token_ids, *completion_token_ids]
    loss_mask = [0] * len(prompt_token_ids) + [1] * len(completion_token_ids)
    return input_ids, loss_mask


def _tool_result_message(tool_call: dict, content: str) -> dict[str, Any]:
    """Build the ``tool`` message that feeds a cached (off-policy) result back to
    the target, paired to the id of the call the target just generated."""
    message: dict[str, Any] = {"role": "tool", "content": content}
    call_id = tool_call.get("id")
    if call_id:
        message["tool_call_id"] = call_id
    return message


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use in filenames."""
    name = re.sub(r'[/\\:*?"<>|]', "_", name)
    name = name.replace(" ", "_")
    return name.strip("._")
