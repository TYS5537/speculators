"""Stdlib-only HS profiler contracts; no torch, model, network, or RNG changes."""

# ruff: noqa: PT009, PT027 -- Also runnable with stdlib unittest.

import io
import json
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_dsv4 import hs_profile as profile


class HiddenStatesProfileTests(unittest.TestCase):
    def setUp(self):
        self.sink = io.StringIO()
        for context in (
            patch.object(profile, "_STATE", None),
            patch.object(profile.sys, "stderr", self.sink),
            patch.dict(os.environ, {"SPECULATORS_HS_PROFILE_EVERY": "0"}),
        ):
            context.start()
            self.addCleanup(context.stop)

    def records(self):
        return [json.loads(line) for line in self.sink.getvalue().splitlines()]

    def test_default_off_does_not_read_clock_inspect_payload_or_change_return(self):
        payload = Mock()

        @profile.profile_sample
        def operation():
            with profile.profile_stage("request_ms"):
                profile.profile_tensors(payload)
                profile.profile_file(payload)
                return payload

        with patch.object(profile, "perf_counter", side_effect=AssertionError):
            self.assertIs(operation(), payload)
        payload.values.assert_not_called()
        payload.stat.assert_not_called()
        self.assertEqual(self.records(), [])

    def test_sampling_is_per_process_bounded_and_does_not_consume_rng(self):
        os.environ["SPECULATORS_HS_PROFILE_EVERY"] = "2"
        rng = random.getstate()

        @profile.profile_sample
        def operation():
            with profile.profile_stage("get_generated_ms"):
                with profile.profile_stage("http_route_ms"):
                    pass
                with profile.profile_stage("http_load_ms"):
                    profile.profile_tensors(
                        {
                            "hs": SimpleNamespace(
                                numel=lambda: 12, element_size=lambda: 2
                            )
                        }
                    )
            return "unchanged"

        for _ in range(301):
            self.assertEqual(operation(), "unchanged")
        records = self.records()
        self.assertEqual(len(records), 151)
        self.assertEqual(random.getstate(), rng)
        last = records[-1]
        self.assertEqual(last["row_count"], 301)
        self.assertEqual(last["stats"]["tensor_bytes"]["latest"], 24)
        self.assertEqual(last["stats"]["sample_total_ms"]["count"], 151)
        self.assertFalse(last["failed"])
        self.assertTrue(
            all(len(item.values) == 128 for item in profile._STATE.summaries.values())
        )
        self.assertLessEqual(
            last["stats"]["http_route_ms"]["latest"],
            last["stats"]["get_generated_ms"]["latest"],
        )
        with patch.object(profile.os, "getpid", return_value=987654321):
            operation()
        self.assertEqual(self.records()[-1]["row_count"], 1)
        self.assertEqual(self.records()[-1]["stats"]["sample_total_ms"]["count"], 1)

    def test_failed_call_keeps_exception_and_context_is_reset(self):
        os.environ["SPECULATORS_HS_PROFILE_EVERY"] = "1"
        failure = ValueError("private prompt and endpoint must not be logged")

        @profile.profile_sample
        def operation():
            with profile.profile_stage("validate_ms"):
                raise failure

        with self.assertRaises(ValueError) as caught:
            operation()
        self.assertIs(caught.exception, failure)
        self.assertTrue(self.records()[0]["failed"])
        self.assertIn("validate_ms", self.records()[0]["stats"])
        self.assertNotIn(str(failure), self.sink.getvalue())
        self.assertIsNone(profile._ACTIVE.get())

    def test_failed_output_sink_never_replaces_return_or_training_exception(self):
        os.environ["SPECULATORS_HS_PROFILE_EVERY"] = "1"
        failing_sink = SimpleNamespace(write=Mock(side_effect=OSError("closed")))
        with patch.object(profile.sys, "stderr", failing_sink):
            self.assertEqual(profile.profile_sample(lambda: 42)(), 42)

            @profile.profile_sample
            def failed():
                raise RuntimeError("original failure")

            with self.assertRaisesRegex(RuntimeError, "original failure"):
                failed()

    def test_summary_quantiles_use_rolling_window_but_max_and_count_are_lifetime(self):
        summary = profile._Summary()
        summary.add(1000)
        for value in range(128):
            record = summary.add(value)
        self.assertEqual(record["count"], 129)
        self.assertEqual(record["max"], 1000)
        self.assertEqual(record["p50_window"], 63)
        self.assertEqual(record["p95_window"], 121)
        self.assertEqual(record["latest"], 127)

    def test_invalid_sampling_setting_disables_diagnostics(self):
        os.environ["SPECULATORS_HS_PROFILE_EVERY"] = "invalid"
        self.assertEqual(profile.profile_sample(lambda: 42)(), 42)
        self.assertEqual(self.records(), [])


if __name__ == "__main__":
    unittest.main()
