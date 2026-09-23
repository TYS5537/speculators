"""Exercise DSV4 sample sharding and worker lifecycle without loading models."""

# ruff: noqa: PT009, PT027 -- Keep this suite runnable with stdlib unittest.

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_eval import parallel as eval_parallel

ROOT = Path(__file__).resolve().parents[2]


class EvalWorkerTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "dsv4_eval_worker_fixture", ROOT / "scripts/evaluate/dspark_offline_eval.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(self.module)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = self.root / "data/group/samples.jsonl"
        dataset.parent.mkdir(parents=True)
        dataset.write_text(
            "".join(json.dumps({"prompt": str(i)}) + "\n" for i in range(20)),
            encoding="utf-8",
        )
        self.records = self.module._load_jsonl(dataset)
        argv = [
            "eval",
            "--verifier-model",
            "fixture-target",
            "--draft-model",
            "fixture-draft",
            "--target-backend",
            "dsv4-vllm",
            "--vllm-endpoint",
            "http://127.0.0.1:1234/v1",
            "--hidden-states-path",
            str(self.root / "hs"),
            "--served-model-name",
            "fixture-alias",
            "--keep-target-hs",
            "--datasets-root",
            str(self.root / "data"),
            "--output-dir",
            str(self.root / "output"),
            "--device",
            "npu:0",
            "--ascend-devices",
            "8,9,10,11,12,13,14,15",
            "--max-samples",
            "11",
            "--max-new-tokens",
            "2048",
            "--target-request-timeout",
            "360",
        ]
        with patch.object(sys, "argv", argv):
            self.args = self.module.parse_args()

    @staticmethod
    def flag(command, name):
        return command[command.index(name) + 1]

    def fake_completed_worker(self, command, *, env):
        index = int(self.flag(command, "--worker-shard-index"))
        count = int(self.flag(command, "--worker-num-shards"))
        self.assertEqual(count, 8)
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], str(index + 8))
        self.assertEqual(env["OPENAI_API_KEY"], "fixture-private-key")
        if self.args.hs_http_endpoint:
            self.assertEqual(env["DSV4_HS_HTTP_TOKEN"], "fixture-hs-token")
            self.assertEqual(
                self.flag(command, "--hs-http-endpoint"), self.args.hs_http_endpoint
            )
            self.assertNotIn("fixture-hs-token", command)
        self.assertNotIn("--ascend-devices", command)  # No recursive spawning.
        for flag, value in (
            ("--device", "npu:0"),
            ("--target-backend", "dsv4-vllm"),
            ("--vllm-endpoint", self.args.vllm_endpoint),
            ("--hidden-states-path", str(self.args.hidden_states_path)),
            ("--served-model-name", self.args.served_model_name),
            ("--dsv4-verification-mode", self.args.dsv4_verification_mode),
            ("--max-samples", str(self.args.max_samples)),
            ("--max-new-tokens", "2048"),
            ("--target-request-timeout", "360.0"),
        ):
            self.assertEqual(self.flag(command, flag), value)
        self.assertIn("--keep-target-hs", command)
        selected = self.module._select_eval_records(
            self.records,
            dataset_name="samples",
            max_samples=self.args.max_samples,
            seed=self.args.seed,
        )
        shard = self.module._shard_records(
            selected, shard_index=index, num_shards=count
        )
        stats = self.module.EvalStats(elapsed_s=1.0)
        for sample_index, _ in shard:
            stats.add_response(
                SimpleNamespace(
                    num_output_tokens=3,
                    proposal_lengths=[2],
                    accepted_draft_lengths=[sample_index % 3],
                    accept_prob_lists=[[0.8, 0.4]],
                    support_accept_rate_lists=[[0.9, 0.5]],
                )
            )
        self.module._write_outputs(
            Path(self.flag(command, "--output-dir")),
            [self.module._summary_row("samples", len(shard), stats)],
            {"samples": [{"source_index": i} for i, _ in shard]},
        )
        return Mock(poll=Mock(return_value=0), wait=Mock(return_value=0))

    def test_eight_workers_merge_uneven_and_empty_shards_in_both_modes(self):
        for mode, endpoint in (
            ("block", None),
            ("reference", None),
            ("block", "http://hs.fixture:8002"),
            ("reference", "http://hs.fixture:8002"),
        ):
            for limit in (4, 11):
                with self.subTest(mode=mode, limit=limit):
                    self.args.dsv4_verification_mode = mode
                    self.args.hs_http_endpoint = endpoint
                    self.args.max_samples = limit
                    with (
                        patch.object(
                            eval_parallel.subprocess,
                            "Popen",
                            side_effect=self.fake_completed_worker,
                        ) as launch,
                        patch.dict(
                            self.module.os.environ,
                            {
                                "ASCEND_RT_VISIBLE_DEVICES": "0,1",
                                "OPENAI_API_KEY": "fixture-private-key",
                                "DSV4_HS_HTTP_TOKEN": "fixture-hs-token",
                            },
                        ),
                    ):
                        self.module.run_ascend_data_parallel(self.args)
                        self.assertEqual(
                            self.module.os.environ["ASCEND_RT_VISIBLE_DEVICES"], "0,1"
                        )
                    self.assertEqual(launch.call_count, 8)
                    row = json.loads(
                        (self.args.output_dir / "summary.json").read_text()
                    )[0]
                    self.assertEqual(row["dataset"], "group/samples")
                    self.assertEqual(row["num_requests"], limit)
                    self.assertEqual(row["num_proposals"], limit)
                    accepted = sum(i % 3 for i in range(1, limit + 1))
                    self.assertAlmostEqual(
                        row["acceptance_length"], 1 + accepted / limit
                    )
                    self.assertEqual(
                        json.loads(row["position_proposed_counts"]), [limit, limit]
                    )
                    artifacts = self.module._load_jsonl(
                        self.args.output_dir / "artifacts/group/samples.jsonl"
                    )
                    self.assertEqual(
                        [item["source_index"] for item in artifacts],
                        list(range(1, limit + 1)),
                    )

    def test_invalid_devices_fail_before_spawning(self):
        for devices in ("", "8,8", "8,", "8,x", "8,-1"):
            with self.subTest(devices=devices):
                self.args.ascend_devices = devices
                with (
                    patch.object(eval_parallel.subprocess, "Popen") as launch,
                    self.assertRaises(ValueError),
                ):
                    self.module.run_ascend_data_parallel(self.args)
                launch.assert_not_called()

    def test_http_options_fail_early_and_metadata_excludes_secret(self):
        self.args.hs_http_endpoint = "http://hs.fixture:8002"
        self.args.hidden_states_path = None
        with (
            patch.dict(self.module.os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "TOKEN"),
        ):
            self.module._prepare_hs_http(self.args, "dsv4-vllm")
        secret = "fixture-hs-token-0123456789abcdef0123456789"
        with patch.dict(self.module.os.environ, {"DSV4_HS_HTTP_TOKEN": secret}):
            self.assertEqual(
                self.module._prepare_hs_http(self.args, "dsv4-vllm"),
                self.args.hs_http_endpoint,
            )
            self.module._write_backend_metadata(
                self.args,
                {
                    "model_path": "fixture-target",
                    "checkpoint_signature": "fixture",
                },
            )
        self.assertEqual(
            self.args.hidden_states_path, self.args.output_dir / "target-hs-downloads"
        )
        metadata = (self.args.output_dir / "eval_backend.json").read_text()
        self.assertNotIn(secret, metadata)
        self.assertEqual(json.loads(metadata)["hs_transport"], "http")
        with self.assertRaisesRegex(ValueError, "requires"):
            self.module._prepare_hs_http(self.args, "hf")

    def test_failed_worker_stops_siblings_without_waiting_for_first_worker(self):
        healthy = Mock(poll=Mock(return_value=None))
        failed = Mock(poll=Mock(return_value=7))
        with (
            patch.object(
                eval_parallel.subprocess,
                "Popen",
                side_effect=[healthy, failed] + [healthy] * 6,
            ),
            patch.object(eval_parallel, "stop_eval_worker") as stop,
            patch.object(eval_parallel.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "worker failures"),
        ):
            self.module.run_ascend_data_parallel(self.args)
        self.assertEqual(stop.call_count, 8)
        healthy.wait.assert_not_called()
        sleep.assert_not_called()
        self.assertFalse((self.args.output_dir / "summary.json").exists())

    def test_partial_start_failure_and_interrupt_stop_started_workers(self):
        for error in (OSError("cannot start worker"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                child = Mock(poll=Mock(return_value=None))
                with (
                    patch.object(
                        eval_parallel.subprocess, "Popen", side_effect=[child, error]
                    ),
                    self.assertRaises(type(error)),
                ):
                    self.module.run_ascend_data_parallel(self.args)
                child.terminate.assert_called_once_with()
                child.wait.assert_called_once_with(timeout=5)
                child.kill.assert_not_called()

    def test_unresponsive_worker_is_killed_after_bounded_grace(self):
        child = Mock(poll=Mock(return_value=None))
        child.wait.side_effect = [subprocess.TimeoutExpired("fixture", 5), 0]
        self.module._stop_eval_worker(child)
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
