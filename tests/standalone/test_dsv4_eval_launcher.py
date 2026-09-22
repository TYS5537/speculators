"""Single-host launcher contracts using fake children and readiness responses."""

# ruff: noqa: PT009, PT027 -- Keep this suite runnable with stdlib unittest.

import io
import json
import math
import signal
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch
from urllib.error import HTTPError, URLError

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4 import eval_launcher as launcher


class LauncherFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.model = self.root / "model"
        self.model.mkdir()
        self.draft = self.root / "draft"
        self.draft.mkdir()
        self.dataset = self.root / "samples.jsonl"
        self.dataset.write_text('{"prompt": "fixture"}\n', encoding="utf-8")
        self.config = {
            "target_hidden_state_format": HS_FORMAT,
            "aux_hidden_state_layer_ids": [1, 11, 21, 30, 40],
            "speculators_config": {"verifier": {"name_or_path": str(self.model)}},
        }
        self.write_draft_config()
        self.report = {
            "model_path": str(self.model),
            "checkpoint_signature": "fixture-checkpoint-signature",
            "config": {
                "model_type": "deepseek_v4",
                "hidden_size": 4096,
                "num_hidden_layers": 43,
                "hc_mult": 4,
                "vocab_size": 129280,
            },
        }
        inspection = patch.object(
            launcher, "inspect_checkpoint", return_value=self.report
        )
        self.inspect = inspection.start()
        self.addCleanup(inspection.stop)

    def write_draft_config(self):
        (self.draft / "config.json").write_text(
            json.dumps(self.config), encoding="utf-8"
        )

    def argv(self, *extra):
        return [
            "--verifier-model",
            str(self.model),
            "--draft-model",
            str(self.draft),
            "--datasets-root",
            str(self.dataset),
            "--hidden-states-path",
            str(self.root / "hs-parent"),
            "--output-dir",
            str(self.root / "output-parent"),
            "--target-devices",
            "0,1,2,3",
            "--eval-device",
            "4",
            *extra,
        ]

    def plan(self, *extra, run_id="fixture-run"):
        return launcher.build_plan(
            launcher.parse_args(self.argv(*extra)), run_id=run_id, port=23456
        )

    @staticmethod
    def flag(command, name):
        return command[command.index(name) + 1]


class PlanningTests(LauncherFixture):
    def test_split_devices_use_block_evaluator_and_loopback_by_default(self):
        plan = self.plan()
        self.assertFalse(plan.shared_device)
        self.assertEqual(plan.target_devices, [0, 1, 2, 3])
        self.assertEqual(plan.eval_device, 4)
        self.assertEqual(plan.eval_devices, [4])
        self.assertNotIn("--ascend-devices", plan.eval_command)
        self.assertEqual(plan.target_env["ASCEND_RT_VISIBLE_DEVICES"], "0,1,2,3")
        self.assertEqual(plan.eval_env["ASCEND_RT_VISIBLE_DEVICES"], "4")
        self.assertEqual(self.flag(plan.target_command, "--host"), "127.0.0.1")
        self.assertEqual(plan.endpoint, "http://127.0.0.1:23456/v1")
        self.assertEqual(self.flag(plan.target_command, "--tensor-parallel-size"), "4")
        self.assertEqual(self.flag(plan.target_command, "--data-parallel-size"), "1")
        self.assertEqual(
            self.flag(plan.target_command, "--pipeline-parallel-size"), "1"
        )
        self.assertEqual(self.flag(plan.target_command, "--max-num-seqs"), "1")
        self.assertIn("--dsv4", plan.target_command)
        self.assertIn("--dsv4-block-verify", plan.target_command)
        self.assertEqual(self.flag(plan.target_command, "--max-logprobs"), "0")
        self.assertEqual(
            self.flag(plan.target_command, "--logprobs-mode"), "raw_logprobs"
        )
        self.assertEqual(self.flag(plan.eval_command, "--target-backend"), "dsv4-vllm")
        self.assertEqual(
            self.flag(plan.eval_command, "--dsv4-verification-mode"), "block"
        )
        self.assertEqual(plan.verification_mode, "block")
        self.assertEqual(plan.public_metadata()["verification_mode"], "block")
        self.assertEqual(self.flag(plan.eval_command, "--device"), "npu:0")
        self.assertNotIn("--measure-base-speedup", plan.eval_command)
        for command in (plan.target_command, plan.eval_command):
            self.assertFalse(set(command) & {"ssh", "scp", "bash", "sh", "pkill"})

    def test_reference_mode_preserves_original_target_transport(self):
        plan = self.plan("--verification-mode", "reference")
        self.assertNotIn("--dsv4-block-verify", plan.target_command)
        self.assertEqual(self.flag(plan.target_command, "--max-logprobs"), "129280")
        self.assertEqual(
            self.flag(plan.eval_command, "--dsv4-verification-mode"), "reference"
        )
        self.assertEqual(plan.public_metadata()["verification_mode"], "reference")
        self.assertEqual(
            plan.public_metadata()["mode"], "single-host-managed-reference"
        )

    def test_eight_eval_devices_share_target_without_changing_tp_or_block_batching(
        self,
    ):
        for mode in ("block", "reference"):
            with self.subTest(mode=mode):
                plan = self.plan(
                    "--eval-devices",
                    "8,9,10,11,12,13,14,15",
                    "--verification-mode",
                    mode,
                )
                self.assertEqual(plan.eval_devices, list(range(8, 16)))
                self.assertIsNone(plan.eval_device)
                self.assertFalse(plan.shared_device)
                self.assertEqual(plan.public_metadata()["eval_num_workers"], 8)
                self.assertEqual(
                    plan.public_metadata()["eval_devices"], list(range(8, 16))
                )
                self.assertEqual(
                    plan.eval_env["ASCEND_RT_VISIBLE_DEVICES"],
                    "8,9,10,11,12,13,14,15",
                )
                self.assertEqual(
                    self.flag(plan.eval_command, "--ascend-devices"),
                    "8,9,10,11,12,13,14,15",
                )
                self.assertEqual(self.flag(plan.eval_command, "--device"), "npu:0")
                self.assertEqual(
                    self.flag(plan.target_command, "--tensor-parallel-size"), "4"
                )
                self.assertEqual(self.flag(plan.target_command, "--max-num-seqs"), "1")

    def test_overlap_checks_every_eval_device_not_only_the_first(self):
        with self.assertRaisesRegex(ValueError, "allow-shared-device"):
            self.plan("--eval-device", "4,3")
        with self.assertRaisesRegex(ValueError, "memory-utilization"):
            self.plan("--eval-device", "4,3", "--allow-shared-device")
        plan = self.plan(
            "--eval-device",
            "4,3",
            "--allow-shared-device",
            "--target-memory-utilization",
            "0.6",
        )
        self.assertTrue(plan.shared_device)
        self.assertEqual(plan.eval_devices, [4, 3])
        self.assertEqual(self.flag(plan.eval_command, "--ascend-devices"), "4,3")

    def test_unknown_verification_mode_is_rejected_before_target_start(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            launcher.parse_args(self.argv("--verification-mode", "automatic"))
        args = launcher.parse_args(self.argv())
        args.verification_mode = "automatic"
        with self.assertRaisesRegex(ValueError, "verification-mode"):
            launcher.build_plan(args)

    def test_overlap_requires_opt_in_and_explicit_memory_budget(self):
        with self.assertRaisesRegex(ValueError, "allow-shared-device"):
            self.plan("--eval-device", "3")
        with self.assertRaisesRegex(ValueError, "memory-utilization"):
            self.plan("--eval-device", "3", "--allow-shared-device")
        plan = self.plan(
            "--eval-device",
            "3",
            "--allow-shared-device",
            "--target-memory-utilization",
            "0.72",
        )
        self.assertTrue(plan.shared_device)
        self.assertEqual(plan.target_memory_utilization, 0.72)
        self.assertEqual(
            self.flag(plan.target_command, "--gpu-memory-utilization"), "0.72"
        )

    def test_rejects_invalid_device_sets_and_tp_mismatch(self):
        for extra in (
            ("--target-devices", ""),
            ("--target-devices", "0,0"),
            ("--target-devices", "0,"),
            ("--target-devices", "0,x"),
            ("--target-devices=-1",),
            ("--eval-device", ""),
            ("--eval-device", "4,4"),
            ("--eval-device", "4,"),
            ("--eval-device", "4,x"),
            ("--eval-device", "4,-1"),
            ("--target-tp-size", "2"),
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.plan(*extra)

    def test_rejects_invalid_budgets_and_timeouts(self):
        for option, values in (
            ("--target-memory-utilization", [0, 1, -0.1, math.nan, math.inf]),
            ("--startup-timeout", [0, -1, math.nan, math.inf]),
            ("--shutdown-timeout", [0, math.inf]),
            ("--target-request-timeout", [0, math.inf]),
            ("--temperature", [-1, math.nan, math.inf]),
            ("--max-new-tokens", [0, -1]),
        ):
            for value in values:
                with (
                    self.subTest(option=option, value=value),
                    self.assertRaises(ValueError),
                ):
                    self.plan(option, str(value))

    def test_rank_environment_is_removed_without_mutating_parent(self):
        inherited = dict.fromkeys(launcher.RANK_ENVIRONMENT, "stale-distributed-value")
        inherited["USER_FIXTURE_SECRET"] = "inherited-secret-do-not-persist"
        with patch.dict(launcher.os.environ, inherited):
            plan = self.plan()
            for name in launcher.RANK_ENVIRONMENT:
                self.assertNotIn(name, plan.target_env)
                self.assertNotIn(name, plan.eval_env)
                self.assertEqual(launcher.os.environ[name], "stale-distributed-value")
        self.assertNotIn(
            "inherited-secret-do-not-persist", json.dumps(plan.public_metadata())
        )

    def test_authentication_stays_in_child_environment_not_metadata_or_argv(self):
        first = self.plan(run_id="one")
        second = self.plan(run_id="two")
        self.assertNotEqual(first.api_key, second.api_key)
        self.assertNotEqual(first.model_name, second.model_name)
        self.assertEqual(first.target_env["VLLM_API_KEY"], first.api_key)
        self.assertEqual(first.eval_env["OPENAI_API_KEY"], first.api_key)
        for serialized in (
            json.dumps(first.public_metadata()),
            repr(first),
            " ".join(first.target_command),
            " ".join(first.eval_command),
        ):
            self.assertNotIn(first.api_key, serialized)

    def test_local_authentication_bypasses_inherited_http_proxy(self):
        inherited = dict(launcher.os.environ)
        inherited.update(NO_PROXY="old.internal", no_proxy="another.internal")
        # Real Windows environ folds case; simulate the Linux parent mapping.
        with patch.object(launcher.os, "environ", inherited):
            plan = self.plan()
        for environment in (plan.target_env, plan.eval_env):
            exclusions = set(environment["NO_PROXY"].split(","))
            self.assertTrue(
                {"127.0.0.1", "localhost", "old.internal", "another.internal"}
                <= exclusions
            )
            self.assertEqual(environment["NO_PROXY"], environment["no_proxy"])

    def test_build_plan_is_read_only_and_each_run_has_isolated_paths(self):
        first = self.plan(run_id="one")
        second = self.plan(run_id="two")
        self.assertNotEqual(first.output_dir, second.output_dir)
        self.assertNotEqual(first.hidden_states_path, second.hidden_states_path)
        for path in (
            first.output_dir,
            second.output_dir,
            first.hidden_states_path,
            second.hidden_states_path,
        ):
            self.assertFalse(path.exists())
            self.assertTrue(path.is_relative_to(self.root))

    def test_existing_run_directory_is_not_adopted_or_removed(self):
        first = self.plan()
        first.output_dir.mkdir(parents=True)
        marker = first.output_dir / "user-owned.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing run"):
            self.plan()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_draft_format_target_and_layer_contract_are_checked(self):
        self.config["target_hidden_state_format"] = "standard"
        self.write_draft_config()
        with self.assertRaisesRegex(ValueError, "DSV4 hidden-state"):
            self.plan()
        self.config["target_hidden_state_format"] = HS_FORMAT
        self.config["speculators_config"]["verifier"]["name_or_path"] = str(
            self.root / "other"
        )
        self.write_draft_config()
        with self.assertRaisesRegex(ValueError, "paths must match"):
            self.plan()
        self.config["speculators_config"]["verifier"]["name_or_path"] = str(self.model)
        self.config["aux_hidden_state_layer_ids"] = [1, 43]
        self.write_draft_config()
        with self.assertRaises(ValueError):
            self.plan()

    def test_dry_run_prints_safe_plan_without_launching_or_making_directories(self):
        output = io.StringIO()
        with (
            patch.object(launcher, "choose_port", return_value=23456),
            patch.object(launcher, "run_plan") as run,
            patch.object(launcher, "start_process") as start,
            patch.object(
                launcher.secrets, "token_urlsafe", return_value="test-private-key"
            ),
            redirect_stdout(output),
        ):
            self.assertEqual(launcher.main(self.argv("--dry-run")), 0)
        run.assert_not_called()
        start.assert_not_called()
        self.assertNotIn("test-private-key", output.getvalue())
        metadata = json.loads(output.getvalue())
        self.assertFalse(Path(metadata["output_dir"]).exists())
        self.assertFalse(Path(metadata["hidden_states_path"]).exists())

    def test_evaluation_controls_are_forwarded_without_changing_algorithm(self):
        plan = self.plan(
            "--temperature",
            "1.0",
            "--seed",
            "123",
            "--max-new-tokens",
            "96",
            "--max-model-len",
            "8192",
            "--datasets",
            "gsm8k,math500",
            "--target-quantization",
            "ascend",
            "--keep-target-hs",
            "--skip-artifacts",
        )
        for flag, value in (
            ("--temperature", "1.0"),
            ("--seed", "123"),
            ("--max-new-tokens", "96"),
            ("--dsv4-max-model-len", "8192"),
            ("--datasets", "gsm8k,math500"),
            ("--target-backend", "dsv4-vllm"),
        ):
            self.assertEqual(self.flag(plan.eval_command, flag), value)
        self.assertIn("--keep-target-hs", plan.eval_command)
        self.assertIn("--skip-artifacts", plan.eval_command)
        self.assertEqual(self.flag(plan.target_command, "--max-model-len"), "8192")
        self.assertEqual(self.flag(plan.target_command, "--quantization"), "ascend")


class ReadinessTests(LauncherFixture):
    def test_waits_for_owned_model_with_matching_alias(self):
        plan = self.plan()
        process = Mock()
        process.poll.return_value = None
        payload = json.dumps({"data": [{"id": plan.model_name}]}).encode()
        with patch.object(
            launcher, "_read_response", side_effect=[b"", payload]
        ) as read:
            launcher.wait_until_ready(process, plan)
        self.assertEqual(
            read.call_args_list[0].args[1], "http://127.0.0.1:23456/health"
        )
        self.assertEqual(read.call_args_list[1].args[1], plan.endpoint + "/models")
        self.assertEqual(read.call_args_list[1].args[2], plan.api_key)

    def test_foreign_ready_model_is_rejected_without_killing_it(self):
        plan = self.plan()
        process = Mock()
        process.poll.return_value = None
        with (
            patch.object(
                launcher,
                "_read_response",
                side_effect=[b"", b'{"data": [{"id": "someone-elses-model"}]}'],
            ),
            patch.object(launcher, "stop_process") as stop,
            self.assertRaisesRegex(RuntimeError, "not this run's target"),
        ):
            launcher.wait_until_ready(process, plan)
        stop.assert_not_called()

    def test_target_exit_preempts_network_readiness(self):
        plan = self.plan()
        process = Mock()
        process.poll.return_value = 7
        with (
            patch.object(launcher, "_read_response") as read,
            self.assertRaisesRegex(RuntimeError, "exited during startup"),
        ):
            launcher.wait_until_ready(process, plan)
        read.assert_not_called()

    def test_readiness_timeout_does_not_start_evaluator(self):
        plan = self.plan()
        plan.startup_timeout = 1
        process = Mock()
        process.poll.return_value = None
        with (
            patch.object(launcher.time, "monotonic", side_effect=[0, 0, 2]),
            patch.object(launcher, "_read_response") as read,
            self.assertRaisesRegex(TimeoutError, "startup timed out"),
        ):
            launcher.wait_until_ready(process, plan)
        read.assert_not_called()

    def test_authentication_failure_is_not_retried_as_model_loading(self):
        plan = self.plan()
        process = Mock()
        process.poll.return_value = None
        error = HTTPError(plan.endpoint, 401, "Unauthorized", {}, None)
        with (
            patch.object(launcher, "_read_response", side_effect=error) as read,
            self.assertRaisesRegex(RuntimeError, "authentication"),
        ):
            launcher.wait_until_ready(process, plan)
        self.assertEqual(read.call_count, 1)

    def test_connection_loading_error_can_recover_to_ready(self):
        plan = self.plan()
        process = Mock()
        process.poll.return_value = None
        payload = json.dumps({"data": [{"id": plan.model_name}]}).encode()
        with (
            patch.object(
                launcher,
                "_read_response",
                side_effect=[URLError("still loading"), b"", payload],
            ) as read,
            patch.object(launcher.time, "sleep"),
        ):
            launcher.wait_until_ready(process, plan)
        self.assertEqual(read.call_count, 3)


class LifecycleTests(LauncherFixture):
    def setUp(self):
        super().setUp()
        for name, replacement in (
            ("_require_posix", lambda: None),
            ("_handle_signals", nullcontext),
        ):
            patcher = patch.object(launcher, name, replacement, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.target = Mock(name="owned_target", pid=101)
        self.target.poll.return_value = None
        self.evaluator = Mock(name="owned_evaluator", pid=102)
        self.evaluator.poll.return_value = 0

    def read_status(self, plan):
        return json.loads(
            (plan.output_dir / "launcher.json").read_text(encoding="utf-8")
        )

    def test_success_starts_target_then_eval_and_cleans_only_owned_children(self):
        plan = self.plan()
        foreign_directory = self.root / "unrelated-service"
        foreign_directory.mkdir()
        marker = foreign_directory / "keep.txt"
        marker.write_text("untouched", encoding="utf-8")
        events = []

        def start(command, **kwargs):
            events.append(("start", command))
            return self.target if len(events) == 1 else self.evaluator

        def ready(process, current_plan):
            self.assertIs(process, self.target)
            self.assertIs(current_plan, plan)
            events.append(("ready", process))

        with (
            patch.object(launcher, "start_process", side_effect=start) as spawn,
            patch.object(launcher, "wait_until_ready", side_effect=ready),
            patch.object(launcher, "ensure_manifest") as manifest,
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 0)
        self.assertEqual([event[0] for event in events], ["start", "ready", "start"])
        self.assertEqual(spawn.call_args_list[0].args[0], plan.target_command)
        self.assertEqual(spawn.call_args_list[1].args[0], plan.eval_command)
        self.assertEqual(spawn.call_args_list[0].kwargs["env"], plan.target_env)
        self.assertEqual(spawn.call_args_list[1].kwargs["env"], plan.eval_env)
        self.assertEqual(
            stop.call_args_list,
            [
                call(self.evaluator, timeout=plan.shutdown_timeout),
                call(self.target, timeout=plan.shutdown_timeout),
            ],
        )
        manifest.assert_called_once()
        self.assertEqual(self.read_status(plan)["status"], "completed")
        self.assertTrue((plan.output_dir / "target.log").exists())
        self.assertTrue((plan.output_dir / "eval.log").exists())
        self.assertEqual(marker.read_text(encoding="utf-8"), "untouched")
        for path in plan.output_dir.iterdir():
            self.assertNotIn(plan.api_key, path.read_text(encoding="utf-8"))

    def test_target_spawn_failure_does_not_stop_an_unowned_process(self):
        plan = self.plan()
        with (
            patch.object(
                launcher, "start_process", side_effect=OSError("cannot spawn")
            ),
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 1)
        stop.assert_not_called()
        self.assertEqual(self.read_status(plan)["status"], "failed")

    def test_startup_failure_cleans_target_and_never_launches_eval(self):
        plan = self.plan()
        with (
            patch.object(launcher, "start_process", return_value=self.target) as spawn,
            patch.object(
                launcher,
                "wait_until_ready",
                side_effect=TimeoutError("loading timeout"),
            ),
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 1)
        self.assertEqual(spawn.call_count, 1)
        stop.assert_called_once_with(self.target, timeout=plan.shutdown_timeout)
        self.assertIn("loading timeout", self.read_status(plan)["error"])

    def test_manifest_mismatch_cleans_target_before_eval_can_start(self):
        plan = self.plan()
        with (
            patch.object(launcher, "start_process", return_value=self.target) as spawn,
            patch.object(launcher, "wait_until_ready"),
            patch.object(
                launcher, "ensure_manifest", side_effect=ValueError("wrong HS contract")
            ),
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 1)
        self.assertEqual(spawn.call_count, 1)
        stop.assert_called_once_with(self.target, timeout=plan.shutdown_timeout)
        self.assertIn("wrong HS contract", self.read_status(plan)["error"])

    def test_evaluator_nonzero_status_is_propagated_after_both_cleanups(self):
        plan = self.plan()
        self.evaluator.poll.return_value = 9
        with (
            patch.object(
                launcher, "start_process", side_effect=[self.target, self.evaluator]
            ),
            patch.object(launcher, "wait_until_ready"),
            patch.object(launcher, "ensure_manifest"),
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 9)
        self.assertEqual(
            [item.args[0] for item in stop.call_args_list],
            [self.evaluator, self.target],
        )
        self.assertEqual(self.read_status(plan)["exit_code"], 9)

    def test_target_exit_during_eval_fails_and_cleans_both_owned_children(self):
        plan = self.plan()
        self.target.poll.return_value = 7
        self.evaluator.poll.return_value = None
        with (
            patch.object(
                launcher, "start_process", side_effect=[self.target, self.evaluator]
            ),
            patch.object(launcher, "wait_until_ready"),
            patch.object(launcher, "ensure_manifest"),
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 1)
        self.assertEqual(
            [item.args[0] for item in stop.call_args_list],
            [self.evaluator, self.target],
        )
        self.assertIn("exited during evaluation", self.read_status(plan)["error"])

    def test_eval_spawn_failure_still_cleans_owned_target(self):
        plan = self.plan()
        with (
            patch.object(
                launcher,
                "start_process",
                side_effect=[self.target, OSError("eval spawn failed")],
            ),
            patch.object(launcher, "wait_until_ready"),
            patch.object(launcher, "ensure_manifest"),
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 1)
        stop.assert_called_once_with(self.target, timeout=plan.shutdown_timeout)
        self.assertIn("eval spawn failed", self.read_status(plan)["error"])

    def test_signal_during_target_spawn_still_registers_and_cleans_owned_target(self):
        plan = self.plan()

        def start(*args, **kwargs):
            plan.interrupted_signal = signal.SIGINT
            return self.target

        with (
            patch.object(launcher, "start_process", side_effect=start) as spawn,
            patch.object(launcher, "wait_until_ready") as ready,
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 128 + signal.SIGINT)
        self.assertEqual(spawn.call_count, 1)
        ready.assert_not_called()
        stop.assert_called_once_with(self.target, timeout=plan.shutdown_timeout)
        self.assertEqual(self.read_status(plan)["status"], "interrupted")

    def test_signal_during_eval_spawn_cleans_both_newly_owned_children(self):
        plan = self.plan()

        def start(command, **kwargs):
            if command == plan.target_command:
                return self.target
            plan.interrupted_signal = signal.SIGTERM
            return self.evaluator

        with (
            patch.object(launcher, "start_process", side_effect=start),
            patch.object(launcher, "wait_until_ready"),
            patch.object(launcher, "ensure_manifest"),
            patch.object(launcher, "stop_process") as stop,
        ):
            self.assertEqual(launcher.run_plan(plan), 128 + signal.SIGTERM)
        self.assertEqual(
            [item.args[0] for item in stop.call_args_list],
            [self.evaluator, self.target],
        )

    def test_first_signal_during_cleanup_does_not_skip_target_cleanup(self):
        plan = self.plan()
        stopped = []

        def stop(process, **kwargs):
            stopped.append(process)
            if process is self.evaluator:
                plan.interrupted_signal = signal.SIGINT

        with (
            patch.object(
                launcher, "start_process", side_effect=[self.target, self.evaluator]
            ),
            patch.object(launcher, "wait_until_ready"),
            patch.object(launcher, "ensure_manifest"),
            patch.object(launcher, "stop_process", side_effect=stop),
        ):
            self.assertEqual(launcher.run_plan(plan), 128 + signal.SIGINT)
        self.assertEqual(stopped, [self.evaluator, self.target])
        self.assertEqual(self.read_status(plan)["status"], "interrupted")


class LocalProcessSmokeTests(LauncherFixture):
    def test_owned_local_http_target_eval_and_cleanup_without_npu(self):
        """Real stdlib children and loopback HTTP, not a model/backend test."""
        plan = launcher.build_plan(
            launcher.parse_args(self.argv()),
            run_id="real-loopback-smoke",
            port=launcher.choose_port(0),
        )
        plan.startup_timeout = 10.0
        plan.shutdown_timeout = 2.0
        manifest = launcher.make_manifest(plan.report, plan.layer_ids)
        manifest["runtime_quantization"] = {"method": plan.target_quantization}
        plan.target_env.update(
            DSV4_TEST_PORT=str(plan.port),
            DSV4_TEST_ALIAS=plan.model_name,
            DSV4_TEST_HS=str(plan.hidden_states_path),
            DSV4_TEST_MANIFEST=json.dumps(manifest),
        )
        plan.eval_env.update(
            DSV4_TEST_ENDPOINT=plan.endpoint,
            DSV4_TEST_ALIAS=plan.model_name,
            DSV4_TEST_RESULT=str(plan.output_dir / "local-smoke-result.txt"),
        )
        target_source = """
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

Path(os.environ['DSV4_TEST_HS'], 'dspark_dsv4_hs.json').write_text(
    os.environ['DSV4_TEST_MANIFEST'], encoding='utf-8'
)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.headers.get('Authorization') != 'Bearer ' + os.environ['VLLM_API_KEY']:
            self.send_response(401)
            self.end_headers()
            return
        if self.path == '/health':
            payload = b'healthy'
        elif self.path == '/v1/models':
            models = {'data': [{'id': os.environ['DSV4_TEST_ALIAS']}]}
            payload = json.dumps(models).encode()
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass

HTTPServer(('127.0.0.1', int(os.environ['DSV4_TEST_PORT'])), Handler).serve_forever()
"""
        eval_source = """
import json
import os
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

request = Request(
    os.environ['DSV4_TEST_ENDPOINT'] + '/models',
    headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']},
)
with build_opener(ProxyHandler({})).open(request, timeout=2.0) as response:
    models = json.loads(response.read())
assert models['data'][0]['id'] == os.environ['DSV4_TEST_ALIAS']
Path(os.environ['DSV4_TEST_RESULT']).write_text(
    'evaluation completed', encoding='utf-8'
)
"""
        plan.target_command = [sys.executable, "-u", "-c", target_source]
        plan.eval_command = [sys.executable, "-u", "-c", eval_source]
        start_owned = launcher.start_process
        children = []

        def start(*args, **kwargs):
            child = start_owned(*args, **kwargs)
            children.append(child)
            return child

        with (
            patch.object(launcher, "_require_posix"),
            patch.object(launcher, "start_process", side_effect=start),
        ):
            code = launcher.run_plan(plan)
        self.assertEqual(code, 0)
        self.assertEqual(len(children), 2)
        self.assertTrue(all(child.poll() is not None for child in children))
        self.assertEqual(
            (plan.output_dir / "local-smoke-result.txt").read_text(encoding="utf-8"),
            "evaluation completed",
        )
        metadata = (plan.output_dir / "launcher.json").read_text(encoding="utf-8")
        self.assertEqual(json.loads(metadata)["status"], "completed")
        self.assertNotIn(plan.api_key, metadata)


if __name__ == "__main__":
    unittest.main()
