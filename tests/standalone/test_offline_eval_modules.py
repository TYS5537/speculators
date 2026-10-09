"""Public module boundaries and legacy entrypoint contracts without model imports."""

# ruff: noqa: PT009 -- Keep this suite runnable with stdlib unittest.

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from speculators_eval import cli, data, loading, parallel, reporting

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/evaluate/dspark_offline_eval.py"


class OfflineEvalModuleTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "offline_eval_boundary_fixture", EVALUATOR
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.module
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(self.module)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def args(self):
        argv = [
            "eval",
            "--verifier-model",
            "local target",
            "--draft-model",
            "local draft",
            "--datasets-root",
            str(self.root / "data"),
            "--output-dir",
            str(self.root / "output"),
        ]
        with patch.object(sys, "argv", argv):
            return self.module.parse_args()

    def test_script_reexports_pure_helpers_without_duplicate_implementations(self):
        for owner, names in (
            (
                loading,
                (
                    "resolve_draft_attn_impl",
                    "parse_bool_override",
                    "prepare_hs_http",
                    "write_backend_metadata",
                ),
            ),
            (
                data,
                (
                    "load_jsonl",
                    "select_eval_records",
                    "prompt_from_record",
                    "discover_datasets",
                    "dataset_id",
                    "split_csv",
                    "shard_records",
                ),
            ),
            (
                reporting,
                (
                    "dataset_output_path",
                    "aggregate_rows",
                    "summary_row",
                    "write_outputs",
                    "read_worker_row",
                    "read_worker_artifacts",
                ),
            ),
        ):
            for name in names:
                with self.subTest(name=name):
                    self.assertIs(
                        getattr(self.module, "_" + name), getattr(owner, name)
                    )
        self.assertIs(self.module.EvalStats, reporting.EvalStats)
        self.assertIs(self.module.RESULT_COLUMNS, reporting.RESULT_COLUMNS)
        self.assertIs(
            self.module.DEEPSPEC_EVAL_SAMPLE_LIMITS, data.DEEPSPEC_EVAL_SAMPLE_LIMITS
        )

    def test_worker_wrapper_supplies_the_original_script(self):
        args = self.args()
        kwargs = {
            "dataset_path": self.root / "data.jsonl",
            "shard_index": 0,
            "num_shards": 2,
            "output_dir": self.root / "shard",
        }
        with patch.object(
            parallel, "worker_command", return_value=["fixture"]
        ) as build:
            self.assertEqual(self.module._worker_command(args, **kwargs), ["fixture"])
        build.assert_called_once_with(args, entrypoint=EVALUATOR, **kwargs)
        with patch.object(parallel, "run_ascend_data_parallel") as run:
            self.module.run_ascend_data_parallel(args)
        run.assert_called_once_with(args, entrypoint=EVALUATOR)

    def test_draft_cache_switch_reaches_hf_and_dsv4_workers(self):
        args = self.args()
        self.assertFalse(args.draft_kv_reuse)
        kwargs = {
            "entrypoint": EVALUATOR,
            "dataset_path": self.root / "data.jsonl",
            "shard_index": 0,
            "num_shards": 2,
            "output_dir": self.root / "shard",
        }
        self.assertNotIn("--draft-kv-reuse", parallel.worker_command(args, **kwargs))
        args.draft_kv_reuse = True
        for backend in ("hf", "dsv4-vllm"):
            args.target_backend = backend
            with self.subTest(backend=backend):
                self.assertEqual(
                    parallel.worker_command(args, **kwargs).count("--draft-kv-reuse"), 1
                )

    def test_explicit_worker_entrypoint_keeps_arguments_and_paths_with_spaces(self):
        args = self.args()
        kwargs = {
            "dataset_path": self.root / "data with spaces.jsonl",
            "shard_index": 1,
            "num_shards": 3,
            "output_dir": self.root / "shard with spaces",
        }
        script_command = self.module._worker_command(args, **kwargs)
        self.assertEqual(script_command[:2], [sys.executable, str(EVALUATOR)])
        entrypoint = self.root / "another entrypoint.py"
        package_command = parallel.worker_command(args, entrypoint=entrypoint, **kwargs)
        self.assertEqual(package_command[1], str(entrypoint))
        self.assertEqual(package_command[2:], script_command[2:])
        self.assertIn(str(kwargs["dataset_path"]), package_command)
        self.assertNotIn("--ascend-devices", package_command)

    def test_data_to_report_round_trip_preserves_weighted_statistics(self):
        dataset = self.root / "datasets/group/math500.jsonl"
        dataset.parent.mkdir(parents=True)
        dataset.write_text(
            "".join(json.dumps({"prompt": str(i)}) + "\n" for i in range(10)),
            encoding="utf-8",
        )
        records = data.select_eval_records(
            data.load_jsonl(dataset), dataset_name=dataset.stem, max_samples=5, seed=7
        )
        shards = [
            data.shard_records(records, shard_index=i, num_shards=3) for i in range(3)
        ]
        self.assertEqual([len(shard) for shard in shards], [2, 2, 1])
        self.assertEqual(
            sorted(index for shard in shards for index, _ in shard), list(range(1, 6))
        )
        rows = []
        for shard in shards:
            stats = reporting.EvalStats(elapsed_s=1.0)
            for index, _ in shard:
                stats.total_output_tokens += 3
                stats.num_proposals += 1
                stats.num_proposed_draft_tokens += 2
                stats.num_accepted_draft_tokens += index % 3
                stats.add_proposal_positions(2, index % 3)
                stats.add_proposal_probability_stats(2, [0.8, 0.4], [0.9, 0.5])
            rows.append(reporting.summary_row(dataset.stem, len(shard), stats))
        identity = data.dataset_id(dataset, self.root / "datasets")
        summary = reporting.aggregate_rows(identity, rows)
        self.assertEqual(summary["num_requests"], 5)
        self.assertEqual(summary["num_proposed_draft_tokens"], 10)
        self.assertEqual(summary["num_accepted_draft_tokens"], 6)
        self.assertAlmostEqual(summary["acceptance_length"], 2.2)
        artifacts = [{"source_index": index} for index in range(1, 6)]
        output = self.root / "output"
        reporting.write_outputs(output, [summary], {identity: artifacts})
        self.assertEqual(reporting.read_worker_row(output), summary)
        self.assertEqual(reporting.read_worker_artifacts(output, identity), artifacts)
        header = (output / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header.split(","), reporting.RESULT_COLUMNS)

    def test_package_and_cli_help_work_without_installed_model_dependencies(self):
        code = textwrap.dedent("""\
            import importlib.abc
            import importlib.util
            import sys

            forbidden = {"torch", "torch_npu", "transformers", "vllm", "speculators"}
            class NoModelImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in forbidden:
                        raise AssertionError("Unexpected model import: " + fullname)
            sys.meta_path.insert(0, NoModelImports())
            sys.path.insert(0, sys.argv[1])
            from speculators_eval import cli, data, loading, orchestration
            from speculators_eval import parallel, reporting
            spec = importlib.util.spec_from_file_location(
                "eval_help_fixture", sys.argv[2]
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            assert module.torch is None
            assert not forbidden.intersection(sys.modules)
            parsed = cli.parse_args([
                "--verifier-model", "target", "--draft-model", "draft",
                "--datasets-root", "data",
            ])
            assert parsed.seed == 980406
            assert parsed.target_backend == "hf"
            assert not forbidden.intersection(sys.modules)
            sys.argv = [sys.argv[2], "--help"]
            module.main()
        """)
        result = subprocess.run(  # noqa: S603 -- Fixed import/help probe, no models.
            [sys.executable, "-S", "-c", code, str(ROOT / "src"), str(EVALUATOR)],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--datasets", "--ascend-devices", "--target-backend"):
            self.assertIn(option, result.stdout)

    def test_cli_defaults_and_explicit_options_match_the_script(self):
        required = [
            "--verifier-model",
            "target",
            "--draft-model",
            "draft",
            "--datasets-root",
            "data",
        ]
        defaults = vars(cli.parse_args(required))
        self.assertEqual(
            defaults,
            {
                "verifier_model": "target",
                "draft_model": "draft",
                "datasets_root": Path("data"),
                "target_backend": "hf",
                "vllm_endpoint": None,
                "hidden_states_path": None,
                "hs_http_endpoint": None,
                "served_model_name": None,
                "dsv4_max_model_len": 4096,
                "dsv4_verification_mode": "reference",
                "dsv4_replay_cache": Path("dsv4_greedy_traces"),
                "dsv4_replay_cache_tag": "",
                "dsv4_replay_audit_samples": 0,
                "dsv4_kv_reuse": False,
                "dsv4_block_output": "auto",
                "dsv4_profile": False,
                "target_request_timeout": 120.0,
                "keep_target_hs": False,
                "datasets": None,
                "output_dir": Path("dspark_offline_eval"),
                "max_samples": None,
                "max_new_tokens": 512,
                "temperature": 0.0,
                "seed": 980406,
                "enable_thinking": "false",
                "raw_prompt_mode": "auto",
                "device": "cuda",
                "dtype": "bfloat16",
                "draft_kv_reuse": False,
                "draft_attn_impl": "auto",
                "no_progress": False,
                "log_every": 10,
                "measure_base_speedup": False,
                "throughput_warmup_samples": 1,
                "skip_artifacts": False,
                "trust_remote_code": False,
                "d2t_path": None,
                "t2d_path": None,
                "sample_from_anchor": None,
                "ascend_devices": None,
                "worker_shard_index": None,
                "worker_num_shards": 1,
                "worker_progress_path": None,
            },
        )
        argv = [
            *required,
            "--target-backend",
            "dsv4-vllm",
            "--dsv4-verification-mode",
            "block",
            "--dsv4-kv-reuse",
            "--draft-kv-reuse",
            "--temperature",
            "1.0",
            "--sample-from-anchor",
            "false",
            "--worker-progress-path",
            "a path.json",
        ]
        with patch.object(sys, "argv", ["eval", *argv]):
            legacy = self.module.parse_args()
        package = cli.parse_args(argv)
        self.assertEqual(vars(legacy), vars(package))
        self.assertEqual(package.worker_progress_path, Path("a path.json"))
        self.assertEqual(package.temperature, 1.0)
        self.assertEqual(package.sample_from_anchor, "false")
        self.assertTrue(package.draft_kv_reuse)
        self.assertTrue(package.dsv4_kv_reuse)


if __name__ == "__main__":
    unittest.main()
