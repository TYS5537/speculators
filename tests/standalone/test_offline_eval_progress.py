"""Aggregate progress without models, network calls or accelerator devices."""

# ruff: noqa: PT009 -- Also runnable with stdlib unittest.

import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from speculators_eval import progress
from speculators_eval.parallel import stop_eval_worker, wait_eval_workers


class ProgressTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        clock = patch.object(progress.time, "monotonic", return_value=0.0)
        self.clock = clock.start()
        self.addCleanup(clock.stop)
        tty = patch.object(progress.sys.stderr, "isatty", return_value=False)
        tty.start()
        self.addCleanup(tty.stop)
        logs = patch.object(progress.logger, "info")
        self.log = logs.start()
        self.addCleanup(logs.stop)

    def tracker(self, total=5, workers=3):
        tracker = progress.ParallelProgress("group/math500", total, self.root, workers)
        self.addCleanup(tracker.close)
        return tracker

    def publish(self, tracker, index, count):
        progress.WorkerProgress(tracker.paths[index]).update(count)

    def test_worker_publishes_atomic_counters_and_disabled_reporting_does_no_io(self):
        path = self.root / "worker.count"
        reporter = progress.WorkerProgress(path)
        for value in (0, 1, 2):
            reporter.update(value)
            self.assertEqual(path.read_text(), str(value))
            self.assertFalse(path.with_suffix(".tmp").exists())
        with patch.object(Path, "write_text") as write:
            progress.WorkerProgress(None).update(10)
        write.assert_not_called()

    def test_reporting_failure_warns_once_without_aborting_samples(self):
        reporter = progress.WorkerProgress(self.root / "worker.count")
        with (
            patch.object(Path, "write_text", side_effect=OSError("disk unavailable")),
            patch.object(progress.logger, "warning") as warning,
        ):
            reporter.update(0)
            reporter.update(1)
        warning.assert_called_once()
        self.assertIsNone(reporter.path)

    def test_uneven_workers_aggregate_while_running_and_completed_workers_fill_limits(
        self,
    ):
        tracker = self.tracker()
        self.assertEqual(tracker.limits, [2, 2, 1])
        self.publish(tracker, 0, 1)
        self.publish(tracker, 1, 1)
        tracker.update([(0, None), (1, None), (2, None)])
        self.assertEqual(tracker.completed, 2)
        tracker.update([(0, 0), (1, None), (2, 0)])
        self.assertEqual(tracker.completed, 4)
        tracker.update([(0, 0), (1, 0), (2, 0)])
        self.assertEqual(tracker.completed, 5)
        self.assertEqual(self.log.call_args.args[2:5], (5, 5, 100.0))

    def test_invalid_missing_and_regressing_counts_do_not_inflate_or_reset_progress(
        self,
    ):
        tracker = self.tracker()
        self.publish(tracker, 0, 1)
        tracker.update([(0, None)])
        for raw in (b"", b"partial", b"-1", b"0", b"99", b"\xff"):
            with self.subTest(raw=raw):
                tracker.paths[0].write_bytes(raw)
                tracker.update([(0, None), (1, None), (2, None)])
                self.assertEqual(tracker.completed, 1)
        self.publish(tracker, 0, 2)
        tracker.update([(0, None)])
        self.assertEqual(tracker.completed, 2)

    def test_nohup_reports_initial_periodic_and_final_counts_with_eta(self):
        tracker = self.tracker(total=10, workers=2)
        self.assertEqual(self.log.call_args.args[-1], "--:--:--")
        self.publish(tracker, 0, 2)
        self.clock.return_value = 9.0
        tracker.update([(0, None)])
        self.assertEqual(self.log.call_count, 1)
        self.clock.return_value = 10.0
        tracker.update([(0, None)])
        self.assertEqual(
            self.log.call_args.args[2:], (2, 10, 20.0, "00:00:10", "00:00:40")
        )
        self.clock.return_value = 11.0
        tracker.update([(0, 0), (1, 0)])
        self.assertEqual(
            self.log.call_args.args[2:], (10, 10, 100.0, "00:00:11", "00:00:00")
        )

    def test_zero_and_empty_shards_have_finite_counts(self):
        for total in (0, 3):
            with self.subTest(total=total):
                tracker = self.tracker(total=total, workers=8)
                tracker.update([(index, 0) for index in range(8)])
                self.assertEqual(tracker.completed, total)
                self.assertEqual(sum(tracker.limits), total)
                self.assertEqual(self.log.call_args.args[-1], "00:00:00")

    def test_failed_worker_is_not_reported_as_completed(self):
        tracker = self.tracker()
        self.publish(tracker, 0, 1)
        tracker.update([(0, 7), (1, None), (2, None)])
        tracker.close()
        self.assertEqual(tracker.completed, 1)
        self.assertEqual(self.log.call_args.args[2:5], (1, 5, 20.0))

    def test_terminal_uses_one_bar_and_only_updates_newly_completed_samples(self):
        bar = Mock()
        with (
            patch.object(progress.sys.stderr, "isatty", return_value=True),
            patch.object(progress, "tqdm", return_value=bar) as create,
        ):
            tracker = self.tracker()
        create.assert_called_once()
        self.assertEqual(create.call_args.kwargs["total"], 5)
        self.publish(tracker, 0, 1)
        tracker.update([(0, None)])
        tracker.update([(0, None)])
        tracker.update([(0, 0), (1, 0), (2, 0)])
        self.assertEqual(
            [call.args[0] for call in bar.update.call_args_list], [1, 0, 4]
        )
        tracker.close()
        bar.close.assert_called_once()
        self.log.assert_not_called()

    def test_missing_tqdm_falls_back_to_logs_even_in_a_terminal(self):
        with (
            patch.object(progress.sys.stderr, "isatty", return_value=True),
            patch.object(progress, "tqdm", None),
        ):
            tracker = self.tracker()
        self.assertIsNone(tracker.bar)
        self.log.assert_called_once()

    @unittest.skipUnless(progress.tqdm is not None, "tqdm is optional")
    def test_real_terminal_bar_renders_count_and_percentage(self):
        stream = io.StringIO()
        with (
            patch.object(progress.sys, "stderr", stream),
            patch.object(stream, "isatty", return_value=True),
        ):
            tracker = self.tracker()
            tracker.update([(0, 0), (1, 0), (2, 0)])
            tracker.close()
        self.assertIn("100%", stream.getvalue())
        self.assertIn("5/5", stream.getvalue())
        self.assertIn("sample", stream.getvalue())

    def test_real_children_report_before_exit_without_loading_models(self):
        tracker = self.tracker(total=4, workers=2)
        code = textwrap.dedent("""\
            import sys
            import time
            from pathlib import Path
            from speculators_eval.progress import WorkerProgress

            path = Path(sys.argv[1])
            reporter = WorkerProgress(path)
            reporter.update(1)
            deadline = time.monotonic() + 5
            while not path.with_suffix('.release').exists():
                if time.monotonic() > deadline:
                    raise SystemExit('parent did not observe partial progress')
                time.sleep(0.01)
            reporter.update(2)
        """)
        children = []
        for index, path in enumerate(tracker.paths):
            child = subprocess.Popen(  # noqa: S603 -- Fixed, model-free fixture.
                [sys.executable, "-S", "-c", code, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.addCleanup(stop_eval_worker, child)
            children.append((index, self.root, child))
        update = tracker.update
        observed_partial = []

        def observe(statuses):
            update(statuses)
            if tracker.completed == 2 and all(code is None for _, code in statuses):
                observed_partial.append(tracker.completed)
                for path in tracker.paths:
                    path.with_suffix(".release").touch()

        with patch.object(tracker, "update", side_effect=observe):
            wait_eval_workers(children, "fixture", progress=tracker)
        self.assertTrue(observed_partial)
        self.assertEqual(tracker.completed, 4)


if __name__ == "__main__":
    unittest.main()
