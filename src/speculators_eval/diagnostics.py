"""Failure-only evaluation artifacts; no model imports, retries or skipped samples."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)
_active_sample: ContextVar[dict | None] = ContextVar("eval_sample", default=None)
_SETTINGS = (
    "seed",
    "max_samples",
    "max_new_tokens",
    "temperature",
    "enable_thinking",
    "raw_prompt_mode",
    "target_backend",
    "dsv4_verification_mode",
    "dsv4_block_output",
    "dsv4_max_model_len",
    "draft_model",
    "verifier_model",
)
_JSON_CONTEXT_CHARS = 160


@contextmanager
def capture_sample_failure(
    *, args, path, dataset, selected_index, record, profiler=None
):
    """Identify the selected record, not its original JSONL line or progress count."""
    sample = {
        "dataset": dataset,
        "dataset_path": str(Path(path).absolute()),
        "selected_index": selected_index,
        "index_note": "1-based AFTER selection/shuffle; NOT the JSONL line number",
        "record": record,
        "worker": {
            "shard_index": getattr(args, "worker_shard_index", None) or 0,
            "num_shards": getattr(args, "worker_num_shards", 1),
            "pid": os.getpid(),
            "ascend_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        },
        "settings": {name: getattr(args, name, None) for name in _SETTINGS},
    }
    token = _active_sample.set(sample)
    try:
        yield
    except Exception as error:
        logger.error(
            "[%s] selected sample #%d failed on shard %s/%s (%s); "
            "evaluation stops without retrying or skipping it",
            dataset,
            selected_index,
            sample["worker"]["shard_index"],
            sample["worker"]["num_shards"],
            type(error).__name__,
        )
        request = sample.get("last_target_request")
        if request is not None:
            logger.error(
                "Last target request ID=%s prefix_tokens=%d failed_during_rpc=%s",
                request["request_id"],
                request["prefix_length"],
                request["failed_during_rpc"],
            )
        if isinstance(error, json.JSONDecodeError):
            logger.error(
                "Invalid JSON at character %d/%d (line %d, column %d)",
                error.pos,
                len(error.doc),
                error.lineno,
                error.colno,
            )
        output_dir = getattr(args, "output_dir", None)
        if output_dir is not None:
            _save_failure(output_dir, sample, error, profiler=profiler)
        raise
    finally:
        _active_sample.reset(token)


@contextmanager
def capture_target_request(*, request, verification_mode, output_mode):
    """Attach only allowlisted request fields; never persist clients or headers."""
    sample = _active_sample.get()
    if sample is None:
        yield
        return
    body = request["extra_body"]
    context = {
        "request_id": body["request_id"],
        "model": request["model"],
        "verification_mode": verification_mode,
        "output_mode": output_mode,
        "prefix_length": len(request["prompt"]),
        "prefix_token_ids": request["prompt"],
        "block_options": body.get("kv_transfer_params", {}).get("dsv4_block_verify"),
        "failed_during_rpc": False,
    }
    sample["last_target_request"] = context
    try:
        yield
    except Exception:
        context["failed_during_rpc"] = True
        raise


def _save_failure(output_dir, sample, error, *, profiler=None):
    """Best effort: a full disk or bad diagnostic must not replace the real error."""
    try:
        directory = _write_failure(output_dir, sample, error, profiler=profiler)
        logger.error(
            "Evaluation failure saved to %s (contains sample/response data; "
            "redact before sharing)",
            directory,
        )
    except Exception as diagnostic_error:  # noqa: BLE001 -- Preserve original error.
        logger.warning(
            "Could not save evaluation failure artifacts (%s); "
            "the original evaluation exception is preserved",
            type(diagnostic_error).__name__,
        )


def _write_failure(output_dir, sample, error, *, profiler=None):
    root = (Path(output_dir) / "errors").resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Locally generated name, never a remote request ID or dataset path. Private
    # directory permissions on POSIX protect response/prompt data in shared runs.
    directory = root / f"sample-{sample['selected_index']:06d}-{uuid4().hex}"
    directory.mkdir(mode=0o700)
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample": sample,
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    if profiler is not None and profiler.enabled:
        payload["worker_dataset_timings"] = profiler.snapshot()
    if isinstance(error, json.JSONDecodeError):
        start = max(0, error.pos - _JSON_CONTEXT_CHARS)
        payload["json_error"] = {
            "message": error.msg,
            "position": error.pos,
            "line": error.lineno,
            "column": error.colno,
            "document_length": len(error.doc),
            "position_unit": "Unicode characters, not UTF-8 bytes",
            "snippet_start": start,
            "snippet": error.doc[start : error.pos + _JSON_CONTEXT_CHARS],
            "response_body_file": "response_body.txt",
            "body_note": (
                "Decoded response text from JSONDecodeError.doc, not wire bytes"
            ),
        }
        (directory / "response_body.txt").write_bytes(
            error.doc.encode("utf-8", errors="surrogatepass")
        )
    (directory / "error.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    return directory
