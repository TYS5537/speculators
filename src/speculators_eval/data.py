"""Dataset discovery, sample selection and prompt formatting for offline eval.

This module is standard-library-only and does not initialize a model backend.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

PROMPT_FIELDS = (
    "prompt",
    "input",
    "question",
    "instruction",
    "text",
    "problem",
    "problem_statement",
    "question_content",
)
# Keep the default offline-evaluation workload aligned with DeepSpec-Ascend's
# top-level eval.py. An explicit --max-samples overrides these per-dataset caps.
DEEPSPEC_EVAL_SAMPLE_LIMITS = {
    "gsm8k": 500,
    "math500": 500,
    "aime25": 30,
    "humaneval": 164,
    "mbpp": 256,
    "livecodebench": 500,
    "mt-bench": 80,
    "alpaca": 500,
    "arena-hard-v2": 500,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(item)
    return records


def _canonical_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    aliases = {
        "aime-25": "aime25",
        "human-eval": "humaneval",
        "live-code-bench": "livecodebench",
        "math-500": "math500",
    }
    return aliases.get(normalized, normalized)


def select_eval_records(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    limit = (
        int(max_samples)
        if max_samples is not None
        else DEEPSPEC_EVAL_SAMPLE_LIMITS.get(_canonical_dataset_name(dataset_name))
    )
    if limit is None or len(records) <= limit:
        return records
    if limit < 0:
        raise ValueError("--max-samples must be >= 0")

    selected = list(records)
    random.Random(int(seed)).shuffle(selected)
    return selected[:limit]


def _string_turns(value: Any) -> list[str] | None:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        turns = [item for item in value if item.strip()]
        return turns or None
    return None


def _messages_from_conversations(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    for item in value:
        if not isinstance(item, dict):
            return None
        raw_role = item.get("from", item.get("role"))
        raw_content = item.get("value", item.get("content"))
        if not isinstance(raw_role, str) or not isinstance(raw_content, str):
            return None
        role = role_map.get(raw_role)
        content = raw_content.strip()
        if role is None or not content:
            return None
        if role == "assistant":
            break
        messages.append({"role": role, "content": content})
    return messages or None


def _chat_template_kwargs(args: argparse.Namespace | None) -> dict[str, Any]:
    if args is None:
        return {}
    enable_thinking = getattr(args, "enable_thinking", "false")
    if enable_thinking == "default":
        return {}
    return {"enable_thinking": enable_thinking == "true"}


def _looks_like_chatml(text: str) -> bool:
    return "<|im_start|>" in text or "<|im_end|>" in text


def _format_raw_prompt(
    prompt: str,
    tokenizer,
    *,
    args: argparse.Namespace | None,
) -> str:
    mode = getattr(args, "raw_prompt_mode", "auto") if args is not None else "auto"
    if mode == "raw" or (mode == "auto" and _looks_like_chatml(prompt)):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **_chat_template_kwargs(args),
    )


def prompt_from_record(
    record: dict[str, Any],
    tokenizer,
    *,
    source: str,
    args: argparse.Namespace | None = None,
) -> str:
    turns = _string_turns(record.get("turns"))
    if turns is not None:
        # Match DeepSpec's DSpark evaluation protocol: rows with `turns` contain
        # user turns, and acceptance eval uses only the first turn.
        return _format_raw_prompt(turns[0], tokenizer, args=args)

    messages = record.get("messages")
    if isinstance(messages, list):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(args),
        )

    messages = _messages_from_conversations(record.get("conversations"))
    if messages is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(args),
        )

    turns = _string_turns(record.get("prompt"))
    if turns is not None:
        return _format_raw_prompt("\n\n".join(turns), tokenizer, args=args)

    instruction = _string_turns(record.get("instruction"))
    if instruction is not None:
        # Alpaca-style records split the request across these two fields.
        # Format once after combining them; `output` is the reference answer.
        turns = instruction + (_string_turns(record.get("input")) or [])
        return _format_raw_prompt("\n\n".join(turns), tokenizer, args=args)

    for field in PROMPT_FIELDS:
        turns = _string_turns(record.get(field))
        if turns is not None:
            return _format_raw_prompt("\n\n".join(turns), tokenizer, args=args)

    keys = ", ".join(sorted(record.keys()))
    supported = ", ".join(["turns", "messages", "conversations", *PROMPT_FIELDS])
    raise ValueError(
        f"{source}: record has no supported prompt field ({supported}); keys=[{keys}]"
    )


def discover_datasets(root: Path, names: list[str] | None) -> list[Path]:
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    if names:
        wanted = set(names)
        paths = [
            path
            for path in paths
            if (
                path.stem in wanted
                or path.name in wanted
                or str(path) in wanted
                or dataset_id(path, root) in wanted
            )
        ]
    if not paths:
        raise FileNotFoundError(f"No JSONL datasets found under {root}")
    return paths


def dataset_id(path: Path, root: Path) -> str:
    """Keep dataset identities unique within a recursively discovered root."""
    # Preserve logical input names, including symlink aliases: workers receive
    # the same logical single-file path and name their artifacts from its stem.
    root = Path(os.path.abspath(root))  # noqa: PTH100 -- Preserve symlink aliases.
    path = Path(os.path.abspath(path))  # noqa: PTH100 -- Preserve symlink aliases.
    if root.is_file():
        if path != root:
            raise ValueError(f"Dataset {path} does not match the input file {root}")
        return path.stem
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Dataset {path} is outside the input root {root}") from exc
    return relative.with_suffix("").as_posix()


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def shard_records(
    records: list[dict[str, Any]],
    *,
    shard_index: int | None,
    num_shards: int,
) -> list[tuple[int, dict[str, Any]]]:
    indexed_records = list(enumerate(records, start=1))
    if shard_index is None or num_shards <= 1:
        return indexed_records
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    return [
        item
        for zero_based_index, item in enumerate(indexed_records)
        if zero_based_index % num_shards == shard_index
    ]
