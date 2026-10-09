"""Opt-in, process-local HS timings with bounded rolling statistics.

This module is stdlib-only so HTTP preflight remains usable without torch.
``SPECULATORS_HS_PROFILE_EVERY=N`` samples every Nth Arrow row per worker.
Nested HTTP stages join the row's one record; they are included in (not added
to) ``get_generated_ms``. Nothing is inserted into training batches or RNGs.
"""

import json
import math
import os
import sys
from collections import deque
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_WINDOW = 128
_ACTIVE = ContextVar("speculators_hs_profile", default=None)
_STATE = None
_NOOP = nullcontext()


class _Summary:
    def __init__(self):
        self.values = deque(maxlen=_WINDOW)
        self.count = 0
        self.maximum = 0.0

    def add(self, value):
        self.values.append(value)
        self.count += 1
        self.maximum = max(self.maximum, value)
        ordered = sorted(self.values)
        return {
            "count": self.count,
            "latest": round(value, 3),
            "p50_window": round(ordered[math.ceil(0.50 * len(ordered)) - 1], 3),
            "p95_window": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
            "max": round(self.maximum, 3),
        }


class _WorkerState:
    def __init__(self, pid):
        self.pid = pid
        try:
            self.every = max(
                0, int(os.environ.get("SPECULATORS_HS_PROFILE_EVERY", "0"))
            )
        except ValueError:
            self.every = 0
        self.rows = 0
        self.summaries = {}

    def emit(self, sample, failed):
        stats = {
            name: self.summaries.setdefault(name, _Summary()).add(value)
            for name, value in sample.items()
        }
        record = {
            "event": "hs_profile",
            "pid": self.pid,
            "rank": os.environ.get("RANK", "0"),
            "row_count": self.rows,
            "every": self.every,
            "window": _WINDOW,
            "failed": failed,
            "stats": stats,
        }
        # A dedicated one-line sink also works in nonzero-rank DataLoader workers.
        # Do not include prompts, tokens, handles, endpoints, or exception text.
        sys.stderr.write(json.dumps(record, separators=(",", ":")) + "\n")


def _worker_state():
    global _STATE  # noqa: PLW0603 -- Reset counters after worker fork/spawn.
    pid = os.getpid()
    if _STATE is None or _STATE.pid != pid:
        _STATE = _WorkerState(pid)
    return _STATE


def profile_sample(function):
    """Sample an entire Arrow read, keeping failures and return values unchanged."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        state = _worker_state()
        state.rows += 1
        if not state.every or (state.rows - 1) % state.every:
            return function(*args, **kwargs)
        sample = {}
        token = _ACTIVE.set(sample)
        started = perf_counter()
        failed = True
        try:
            result = function(*args, **kwargs)
            failed = False
            return result
        finally:
            sample["sample_total_ms"] = (perf_counter() - started) * 1000
            _ACTIVE.reset(token)
            # Diagnostics never fail training, including a broken output sink.
            with suppress(Exception):
                state.emit(sample, failed)

    return wrapped


@contextmanager
def _timed_stage(sample, name):
    started = perf_counter()
    try:
        yield
    finally:
        sample[name] = sample.get(name, 0.0) + (perf_counter() - started) * 1000


def profile_stage(name):
    """Return a no-op without reading a clock for disabled/unsampled rows."""
    sample = _ACTIVE.get()
    return _NOOP if sample is None else _timed_stage(sample, name)


def profile_tensors(payload):
    """Record logical tensor bytes only when the enclosing row was sampled."""
    sample = _ACTIVE.get()
    if sample is not None:
        with suppress(Exception):
            sample["tensor_bytes"] = sum(
                tensor.numel() * tensor.element_size() for tensor in payload.values()
            )


def profile_file(path):
    """Record downloaded bytes while the temporary artifact still exists."""
    sample = _ACTIVE.get()
    if sample is not None:
        with suppress(OSError):
            sample["http_file_bytes"] = path.stat().st_size
