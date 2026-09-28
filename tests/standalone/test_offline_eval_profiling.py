"""Diagnostic timers and worker aggregation need no model or accelerator imports."""

# ruff: noqa: PT009, PT027 -- Also runnable with stdlib unittest.

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_eval import profiling


class EvalProfilingTests(unittest.TestCase):
    def test_disabled_profiler_does_not_read_clock_or_synchronize(self):
        profiler = profiling.EvaluationProfiler(device="npu:0")
        with (
            patch.object(profiling.time, "perf_counter") as clock,
            patch.object(profiler, "_synchronize") as sync,
        ):
            with profiler.measure("unused"):
                profiler.count("bytes", 17)
            clock.assert_not_called()
            sync.assert_not_called()
        self.assertEqual(profiler.snapshot(), {"stages": {}, "counters": {}})

    def test_timing_sync_boundaries_and_reset(self):
        profiler = profiling.EvaluationProfiler(enabled=True, device="npu:0")
        with (
            patch.object(
                profiling.time, "perf_counter", side_effect=[1.0, 3.0, 4.0, 9.0]
            ),
            patch.object(profiler, "_synchronize") as sync,
        ):
            with profiling.profile_stage(SimpleNamespace(profiler=profiler), "draft"):
                pass
            with profiler.measure("rpc", synchronize=False):
                profiler.count("bytes", 32)
            self.assertEqual(sync.call_count, 2)
        snapshot = profiler.snapshot()
        self.assertEqual(snapshot["stages"]["draft"], {"seconds": 2.0, "calls": 1})
        self.assertEqual(snapshot["stages"]["rpc"], {"seconds": 5.0, "calls": 1})
        profiler.reset()
        self.assertEqual(profiler.stages, {})
        self.assertEqual(snapshot["counters"]["bytes"], 32)

    def test_failure_is_timed_and_propagated(self):
        profiler = profiling.EvaluationProfiler(enabled=True, device="cpu")
        with (
            self.assertRaisesRegex(RuntimeError, "request failed"),
            profiler.measure("rpc"),
        ):
            raise RuntimeError("request failed")
        self.assertEqual(profiler.stages["rpc"]["calls"], 1)

    def test_worker_reports_sum_work_not_wall_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workers = []
            for i in range(3):
                directory = root / f"worker{i}"
                profiler = profiling.EvaluationProfiler(enabled=True)
                if i != 2:  # Includes an empty shard.
                    profiler.record("target_rpc", i + 2.0)
                    profiler.count("target_packet_bytes", 100)
                profiling.write_timings(directory, {"samples": profiler.snapshot()})
                workers.append((i, directory, Mock()))
            reports = {}
            profiling.collect_worker_timings(
                root, workers, "group/samples", reports, enabled=True
            )
            payload = json.loads((root / "timing.json").read_text())
            result = payload["datasets"]["group/samples"]
            self.assertEqual(
                result["stages"]["target_rpc"], {"seconds": 5.0, "calls": 2}
            )
            self.assertEqual(result["counters"]["target_packet_bytes"], 200)
            self.assertIn("NOT dataset wall time", payload["note"])


if __name__ == "__main__":
    unittest.main()
