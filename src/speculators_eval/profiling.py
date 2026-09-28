"""Opt-in, synchronized evaluation timings; never import Torch in parent workers."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager, nullcontext

logger = logging.getLogger(__name__)


class EvaluationProfiler:
    def __init__(self, *, enabled=False, device=None):
        self.enabled = enabled
        self.device = device
        self.reset()

    def reset(self):
        self.stages = {}
        self.counters = {}

    def _synchronize(self):
        if self.device is None or str(self.device).split(":")[0] == "cpu":
            return
        import torch  # noqa: PLC0415 -- No device initialization in parent workers.

        backend = getattr(torch, str(self.device).split(":")[0], None)
        synchronize = getattr(backend, "synchronize", None)
        if synchronize is not None:
            synchronize(self.device)

    @contextmanager
    def measure(self, name, *, synchronize=True):
        if not self.enabled:
            yield
            return
        if synchronize:
            self._synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize:
                self._synchronize()
            self.record(name, time.perf_counter() - started)

    def record(self, name, seconds):
        if self.enabled:
            stage = self.stages.setdefault(name, {"seconds": 0.0, "calls": 0})
            stage["seconds"] += seconds
            stage["calls"] += 1

    def count(self, name, value):
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + value

    def snapshot(self):
        return {
            "stages": {name: dict(stage) for name, stage in self.stages.items()},
            "counters": dict(self.counters),
        }

    def log(self, dataset):
        if self.enabled:
            logger.info(
                "[%s] timing totals (nested, not additive): %s | %s",
                dataset,
                ", ".join(
                    f"{name}={value['seconds']:.2f}s/{value['calls']} calls"
                    for name, value in self.stages.items()
                ),
                self.counters,
            )


def profile_stage(owner, name):
    profiler = getattr(owner, "profiler", None)
    return profiler.measure(name) if profiler is not None else nullcontext()


def write_timings(output_dir, datasets):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "note": (
            "Diagnostic timings synchronize devices and include overhead. "
            "Stages overlap: generation includes draft/target; server stages are "
            "inside target_rpc, which also includes queueing, export and network. "
            "Batched server_forward/server_head times are equal per-request "
            "shares; their calls count profiled requests, not native forwards. "
            "server_packet_prepare excludes file publication. Worker totals are "
            "summed work time, NOT dataset wall time or serving throughput. "
            "Model loading is outside these timings."
        ),
        "datasets": datasets,
    }
    (output_dir / "timing.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def merge_worker_timings(output_dirs, dataset):
    result = {"stages": {}, "counters": {}}
    for output_dir in output_dirs:
        payload = json.loads((output_dir / "timing.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or len(payload["datasets"]) != 1:
            raise ValueError(f"Invalid worker timing report: {output_dir}")
        # Child roots are single files, so nested dataset IDs become flat stems.
        profile = next(iter(payload["datasets"].values()))
        for name, stage in profile["stages"].items():
            total = result["stages"].setdefault(name, {"seconds": 0.0, "calls": 0})
            total["seconds"] += stage["seconds"]
            total["calls"] += stage["calls"]
        for name, count in profile["counters"].items():
            result["counters"][name] = result["counters"].get(name, 0) + count
    logger.info("[%s] aggregated diagnostic timings from all workers", dataset)
    return result


def collect_worker_timings(output_dir, workers, dataset, profiles, *, enabled):
    if enabled:
        profiles[dataset] = merge_worker_timings(
            [directory for _, directory, _ in workers], dataset
        )
        write_timings(output_dir, profiles)
