"""Failure artifacts remain local and never change evaluation control flow."""

# ruff: noqa: PT009, PT027 -- Also runs with stdlib unittest.

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speculators_eval import diagnostics
from speculators_eval.profiling import EvaluationProfiler


class OfflineEvalDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(
            output_dir=self.root / "eval",
            worker_shard_index=7,
            worker_num_shards=16,
            seed=980406,
            temperature=0.0,
            api_key="do-not-save-this-key",
        )
        self.record = {"instruction": "private prompt \u793a\u4f8b", "input": "test"}
        self.request = {
            "model": "fixture-target",
            "prompt": [1, 2, 3],
            "extra_headers": {"Authorization": "Bearer do-not-save-header"},
            "extra_body": {
                "request_id": "../never-use-this-as-a-path",
                "kv_transfer_params": {
                    "dsv4_block_verify": {
                        "version": 2,
                        "logits_start": 1,
                        "hidden_start": 1,
                        "output_mode": "greedy",
                        "profile": False,
                    }
                },
            },
        }

    def sample(self, index=8, *, profiler=None):
        return diagnostics.capture_sample_failure(
            args=self.args,
            path=self.root / "alpaca.jsonl",
            dataset="alpaca",
            selected_index=index,
            record=self.record,
            profiler=profiler,
        )

    def rpc(self):
        return diagnostics.capture_target_request(
            request=self.request, verification_mode="block", output_mode="greedy"
        )

    def errors(self):
        return sorted(self.args.output_dir.glob("errors/*/error.json"))

    def test_success_has_no_artifacts_or_rng_changes(self):
        state = random.getstate()
        with self.sample(), self.rpc():
            pass
        self.assertFalse(self.args.output_dir.exists())
        self.assertEqual(random.getstate(), state)
        self.assertIsNone(diagnostics._active_sample.get())

    def test_bad_json_saves_exact_decoded_text_and_allowlisted_request(self):
        body = '{\r\n"text":"\u793a\u4f8b", "score":nan}'
        with self.assertRaises(json.JSONDecodeError) as caught:
            json.loads(body)
        error = caught.exception
        with (
            self.assertLogs(diagnostics.logger, level="ERROR") as logs,
            self.assertRaises(json.JSONDecodeError) as saved,
            self.sample(),
            self.rpc(),
        ):
            raise error
        self.assertIs(saved.exception, error)
        (path,) = self.errors()
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            (path.parent / "response_body.txt").read_bytes(), body.encode()
        )
        details = payload["json_error"]
        self.assertEqual(details["position"], body.index("nan"))
        self.assertEqual(details["line"], 2)
        self.assertEqual(details["document_length"], len(body))
        self.assertEqual(details["snippet"], body)
        sample = payload["sample"]
        self.assertEqual(sample["record"], self.record)
        self.assertEqual(sample["selected_index"], 8)
        self.assertEqual(sample["worker"]["shard_index"], 7)
        self.assertEqual(sample["worker"]["num_shards"], 16)
        request = sample["last_target_request"]
        self.assertEqual(request["prefix_token_ids"], [1, 2, 3])
        self.assertEqual(request["prefix_length"], 3)
        self.assertEqual(request["output_mode"], "greedy")
        self.assertTrue(request["failed_during_rpc"])
        self.assertEqual(request["request_id"], "../never-use-this-as-a-path")
        self.assertEqual(path.parent.parent, self.args.output_dir / "errors")
        saved_text = path.read_text(encoding="utf-8")
        for secret in (self.args.api_key, "Bearer do-not-save-header"):
            self.assertNotIn(secret, saved_text)
            self.assertNotIn(secret, str(logs.output))
        self.assertNotIn(self.record["instruction"], str(logs.output))
        self.assertNotIn(body, str(logs.output))
        self.assertIn(str(path.parent), "\n".join(logs.output))
        self.assertIsNone(diagnostics._active_sample.get())

    def test_failure_after_rpc_is_not_mislabelled_as_rpc_failure(self):
        with self.assertRaisesRegex(ValueError, "bad packet"), self.sample():
            with self.rpc():
                pass
            raise ValueError("bad packet")
        (path,) = self.errors()
        payload = json.loads(path.read_text())
        self.assertFalse(payload["sample"]["last_target_request"]["failed_during_rpc"])
        self.assertNotIn("json_error", payload)
        self.assertFalse((path.parent / "response_body.txt").exists())

    def test_requests_do_not_leak_between_samples_and_paths_are_unique(self):
        for i in range(2):
            with self.assertRaises(RuntimeError), self.sample():
                if i == 0:
                    with self.rpc():
                        pass
                raise RuntimeError("failed")
        paths = self.errors()
        self.assertEqual(len(paths), 2)
        payloads = [json.loads(path.read_text())["sample"] for path in paths]
        self.assertEqual(sum("last_target_request" in item for item in payloads), 1)
        self.assertIsNone(diagnostics._active_sample.get())

    def test_diagnostic_write_failure_does_not_mask_original_exception(self):
        failure = json.JSONDecodeError("empty", "", 0)
        with (
            patch.object(
                diagnostics, "_write_failure", side_effect=OSError("disk full")
            ),
            self.assertLogs(diagnostics.logger, level="WARNING") as logs,
            self.assertRaises(json.JSONDecodeError) as caught,
            self.sample(),
        ):
            raise failure
        self.assertIs(caught.exception, failure)
        self.assertIn("Could not save", str(logs.output))
        self.assertFalse(self.args.output_dir.exists())

    def test_unconfigured_output_and_keyboard_interrupt_do_not_write(self):
        with self.assertRaises(KeyboardInterrupt), self.sample():
            raise KeyboardInterrupt
        self.assertFalse(self.args.output_dir.exists())
        del self.args.output_dir
        with (
            patch.object(diagnostics, "_save_failure") as save,
            self.assertRaises(RuntimeError),
            self.sample(),
        ):
            raise RuntimeError("failed")
        save.assert_not_called()
        self.assertIsNone(diagnostics._active_sample.get())

    def test_request_without_sample_is_a_noop(self):
        with self.rpc():
            pass
        self.assertFalse(self.args.output_dir.exists())
        self.assertIsNone(diagnostics._active_sample.get())

    def test_opt_in_timings_survive_an_unfinished_dataset(self):
        profiler = EvaluationProfiler(enabled=True)
        profiler.count("samples", 7)
        profiler.record("target_rpc", 3.0)
        with self.assertRaises(RuntimeError), self.sample(profiler=profiler):
            raise RuntimeError("failed")
        (path,) = self.errors()
        payload = json.loads(path.read_text())
        self.assertEqual(payload["worker_dataset_timings"], profiler.snapshot())


if __name__ == "__main__":
    unittest.main()
