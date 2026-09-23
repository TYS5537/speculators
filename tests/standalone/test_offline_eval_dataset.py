"""Dataset execution contracts with deterministic clocks and no model imports."""

# ruff: noqa: PT009, PT027 -- Also runs with stdlib unittest.

import importlib.util
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_eval import reporting

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/evaluate/dspark_offline_eval.py"


class _DatasetRun:
    def __init__(self, evaluator, *, count=5, paired=False, **options):
        self.evaluator = evaluator
        self.args = SimpleNamespace(
            datasets_root=ROOT / "datasets",
            max_samples=None,
            seed=17,
            throughput_warmup_samples=1,
            no_progress=False,
            skip_artifacts=False,
            log_every=2,
        )
        vars(self.args).update(options)
        self.path = self.args.datasets_root / "group/custom.jsonl"
        self.records = [{"prompt": f"prompt-{i}"} for i in range(1, count + 1)]
        self.stop_ids = [2, 7]
        self.events = []
        self.calls = {}
        self.now = 0.0
        self.costs = {
            "prompt": 3.0,
            "draft": 2.0,
            "base": 4.0,
            "sync": 0.5,
            "artifact": 5.0,
            "log": 7.0,
            "progress": 11.0,
        }
        self.fail_at = None
        self.failure = RuntimeError("injected dataset failure")
        self.output_tokens = {"draft": 4, "base": 6}
        self.runner = self.make_runner("draft")
        self.base_runner = self.make_runner("base") if paired else None
        self.progress_available = True
        self.stats = reporting.EvalStats()
        self.add_response = self.stats.add_response
        self.stats.add_response = self.record_response
        self.selection = Mock(wraps=evaluator._select_eval_records)

    def record(self, name, *details):
        self.events.append((name, details))
        self.now += self.costs.get(name, 0.0)
        self.calls[name] = self.calls.get(name, 0) + 1
        if (name, self.calls[name]) == self.fail_at:
            raise self.failure

    def details(self, name):
        return [details for event, details in self.events if event == name]

    @property
    def names(self):
        return [name for name, _ in self.events]

    def clock(self):
        self.record("clock", self.now)
        return self.now

    def make_runner(self, name):
        def generate(prompt, stop_ids):
            self.record(name, prompt, stop_ids)
            return SimpleNamespace(
                num_input_tokens=2,
                num_output_tokens=self.output_tokens[name],
                output_ids=[SimpleNamespace(tolist=lambda: self.artifact(prompt))],
                proposal_lengths=[3, 1],
                accepted_draft_lengths=[2, 0],
                accept_prob_lists=[[0.5, 0.25, 0.0], [1.0]],
                support_accept_rate_lists=[[0.75, 0.5, 0.25], [0.5]],
            )

        return SimpleNamespace(device=name, tokenizer=object(), generate_one=generate)

    def artifact(self, prompt):
        self.record("artifact", prompt)
        return [7, 8, 9, 10, 11, 12]

    def prompt(self, record, tokenizer, *, source, args):
        self.record("prompt", record, tokenizer, source, args)
        return record["prompt"]

    def progress(self, records, **kwargs):
        self.record("progress", records, kwargs)
        return records

    def record_response(self, response):
        self.record("stats", response)
        self.add_response(response)

    def run(self):
        patches = {
            "_load_jsonl": Mock(return_value=self.records),
            "_select_eval_records": self.selection,
            "_prompt_from_record": self.prompt,
            "_synchronize_device": lambda device: self.record("sync", device),
            "time": SimpleNamespace(perf_counter=self.clock),
            "torch": SimpleNamespace(
                manual_seed=lambda seed: self.record("seed", seed)
            ),
            "tqdm": self.progress if self.progress_available else None,
            "logger": SimpleNamespace(info=lambda *args: self.record("log", *args)),
            "EvalStats": lambda: self.stats,
        }
        with ExitStack() as stack:
            for name, value in patches.items():
                stack.enter_context(patch.object(self.evaluator, name, value))
            return self.evaluator._evaluate_dataset(
                path=self.path,
                runner=self.runner,
                base_runner=self.base_runner,
                args=self.args,
                stop_token_ids=self.stop_ids,
            )


class OfflineEvalDatasetTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "offline_eval_dataset_fixture", EVALUATOR
        )
        self.evaluator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.evaluator
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(self.evaluator)

    def assert_prompt_contract(self, case, indices):
        prompts = case.details("prompt")
        self.assertEqual([entry[2] for entry in prompts], indices)
        for _, tokenizer, _, args in prompts:
            self.assertIs(tokenizer, case.runner.tokenizer)
            self.assertIs(args, case.args)
        for name in ("draft", "base"):
            for _, stop_ids in case.details(name):
                self.assertIs(stop_ids, case.stop_ids)

    def test_unpaired_time_includes_prompt_artifacts_progress_and_logging(self):
        case = _DatasetRun(self.evaluator)
        row, artifacts = case.run()
        self.assertEqual(
            case.details("clock"), [(0.0,), (21.0,), (38.0,), (65.0,), (82.0,), (89.0,)]
        )
        self.assertEqual(case.names[:2], ["clock", "progress"])
        self.assertEqual(case.names[-3:], ["clock", "log", "clock"])
        self.assertNotIn("seed", case.names)
        self.assertNotIn("sync", case.names)
        self.assertEqual(row["elapsed_s"], 89.0)
        self.assertEqual(row["requests_per_second"], 5 / 89.0)
        self.assertEqual(row["output_tokens_per_second"], 20 / 89.0)
        self.assertEqual(row["total_output_tokens"], 20)
        self.assertEqual(row["num_requests"], 5)
        self.assertEqual(row["dataset"], "group/custom")
        self.assertEqual(list(row), reporting.RESULT_COLUMNS)
        self.assertEqual(row["num_proposals"], 10)
        self.assertEqual(row["num_proposed_draft_tokens"], 20)
        self.assertEqual(row["num_accepted_draft_tokens"], 10)
        self.assertEqual(row["acceptance_length"], 2.0)
        self.assertEqual(row["draft_length"], 2.0)
        self.assertEqual(row["accepted_draft_length"], 1.0)
        for key, expected in {
            "position_proposed_counts": [10, 5, 5],
            "position_accepted_counts": [5, 5, 0],
            "position_accept_rates": [0.5, 1.0, 0.0],
            "position_accept_prob_sums": [7.5, 1.25, 0.0],
            "position_support_accept_rate_sums": [6.25, 2.5, 1.25],
            "position_accept_prob_means": [0.75, 0.25, 0.0],
            "position_support_accept_rate_means": [0.625, 0.5, 0.25],
        }.items():
            self.assertEqual(row[key], json.dumps(expected))
        self.assertEqual(
            case.details("progress")[0][1],
            {"total": 5, "desc": "group/custom", "unit": "sample"},
        )
        self.assertEqual(
            case.details("log"),
            [
                (
                    "[%s] %d/%d samples | out_tok=%d | tok/s=%.2f | acc_len=%.3f",
                    "group/custom",
                    processed,
                    5,
                    processed * 4,
                    processed * 4 / elapsed,
                    2.0,
                )
                for processed, elapsed in ((1, 21.0), (2, 38.0), (4, 65.0), (5, 82.0))
            ],
        )
        self.assertEqual(
            artifacts,
            [
                {
                    "prompt": f"prompt-{i}",
                    "output_token_ids": [7, 8, 9, 10, 11, 12],
                    "num_input_tokens": 2,
                    "source_index": i,
                }
                for i in range(1, 6)
            ],
        )
        self.assert_prompt_contract(case, [f"{case.path}:{i}" for i in range(1, 6)])

    def test_paired_warmup_seed_and_synchronized_generation_order(self):
        case = _DatasetRun(self.evaluator, count=3, paired=True)
        row, artifacts = case.run()
        warmup = ["log", "prompt", "draft", "base", "sync", "seed", "clock", "progress"]
        measured = [
            "prompt",
            "sync",
            "clock",
            "draft",
            "sync",
            "clock",
            "sync",
            "clock",
            "base",
            "sync",
            "clock",
            "stats",
            "artifact",
            "log",
        ]
        self.assertEqual(case.names, warmup + measured * 3)
        self.assertEqual(case.details("seed"), [(17,)])
        self.assertEqual(
            case.details("sync"),
            [("draft",)] + [("draft",), ("draft",), ("base",), ("base",)] * 3,
        )
        self.assertEqual(case.details("clock")[0], (16.5,))
        self.assertEqual(row["elapsed_s"], 7.5)
        self.assertEqual(row["base_elapsed_s"], 13.5)
        self.assertEqual(row["total_output_tokens"], 12)
        self.assertEqual(row["base_total_output_tokens"], 18)
        self.assertEqual(row["num_proposals"], 6)
        self.assertEqual(row["output_tokens_per_second"], 12 / 7.5)
        self.assertEqual(row["base_output_tokens_per_second"], 18 / 13.5)
        self.assertAlmostEqual(row["speedup_vs_base"], 1.2)
        self.assertEqual(len(artifacts), 3)
        self.assertEqual(
            case.details("log")[1:],
            [
                (
                    "[%s] %d/%d samples | DSpark=%.2f tok/s | "
                    "base=%.2f tok/s | speedup=%.3fx | acc_len=%.3f",
                    "group/custom",
                    i,
                    3,
                    12 / 7.5,
                    18 / 13.5,
                    (12 / 7.5) / (18 / 13.5),
                    2.0,
                )
                for i in (1, 2, 3)
            ],
        )
        indices = [1, 1, 2, 3]
        self.assert_prompt_contract(case, [f"{case.path}:{i}" for i in indices])
        self.assertEqual(case.details("draft"), case.details("base"))

    def test_selection_and_sharding_precede_warmup_and_keep_source_indices(self):
        case = _DatasetRun(
            self.evaluator,
            count=8,
            paired=True,
            max_samples=5,
            worker_shard_index=1,
            worker_num_shards=3,
            throughput_warmup_samples=8,
        )
        row, artifacts = case.run()
        case.selection.assert_called_once_with(
            case.records, dataset_name="custom", max_samples=5, seed=17
        )
        self.assertEqual(row["num_requests"], 2)
        self.assertEqual([item["source_index"] for item in artifacts], [2, 5])
        self.assertEqual(
            case.details("log")[:2],
            [
                ("[%s] selected %d/%d samples with seed=%d", "group/custom", 5, 8, 17),
                (
                    "[%s] warming up DSpark and base model with %d sample(s)",
                    "group/custom",
                    2,
                ),
            ],
        )
        self.assert_prompt_contract(case, [f"{case.path}:{i}" for i in (2, 5, 2, 5)])
        self.assertEqual(case.details("draft")[:2], case.details("draft")[2:])

    def test_empty_selection_or_shard_still_resets_seed_only_for_paired_runs(self):
        for paired in (False, True):
            for options in (
                {"count": 0},
                {"max_samples": 0},
                {"count": 1, "worker_shard_index": 2, "worker_num_shards": 3},
            ):
                with self.subTest(paired=paired, options=options):
                    case = _DatasetRun(
                        self.evaluator, paired=paired, no_progress=True, **options
                    )
                    row, artifacts = case.run()
                    self.assertEqual(artifacts, [])
                    self.assertEqual(row["num_requests"], 0)
                    self.assertEqual(row["acceptance_length"], 1.0)
                    self.assertEqual(row["elapsed_s"], 0.0)
                    self.assertIs(type(row["output_tokens_per_second"]), int)
                    self.assertEqual(case.details("seed"), [(17,)] if paired else [])
                    self.assertEqual(
                        case.details("sync"), [("draft",)] if paired else []
                    )
                    self.assertEqual(case.calls["clock"], 1 if paired else 2)
                    self.assertNotIn("prompt", case.names)
                    self.assertNotIn("draft", case.names)
                    self.assertNotIn("base", case.names)

    def test_warmup_zero_and_invalid_values_keep_existing_validation_scope(self):
        zero = _DatasetRun(
            self.evaluator, count=1, paired=True, throughput_warmup_samples=0
        )
        zero.run()
        self.assertEqual(zero.names[:4], ["sync", "seed", "clock", "progress"])
        for value in (-1, "invalid", None):
            with self.subTest(value=value):
                single = _DatasetRun(
                    self.evaluator, count=1, throughput_warmup_samples=value
                )
                paired = _DatasetRun(
                    self.evaluator,
                    count=1,
                    paired=True,
                    throughput_warmup_samples=value,
                )
                if value is None:
                    del single.args.throughput_warmup_samples
                    del paired.args.throughput_warmup_samples
                single.run()
                self.assertNotIn("seed", single.names)
                expected_error = AttributeError if value is None else ValueError
                with self.assertRaises(expected_error):
                    paired.run()
                self.assertEqual(paired.events, [])

    def test_progress_and_artifacts_can_be_disabled_independently(self):
        for no_progress, available in ((False, True), (True, True), (False, False)):
            for skip_artifacts in (False, True):
                with self.subTest(
                    no_progress=no_progress, available=available, skip=skip_artifacts
                ):
                    case = _DatasetRun(
                        self.evaluator,
                        count=1,
                        no_progress=no_progress,
                        skip_artifacts=skip_artifacts,
                    )
                    case.progress_available = available
                    row, artifacts = case.run()
                    self.assertEqual(
                        "progress" in case.names, available and not no_progress
                    )
                    self.assertEqual("artifact" in case.names, not skip_artifacts)
                    self.assertEqual(len(artifacts), 0 if skip_artifacts else 1)
                    self.assertEqual(row["total_output_tokens"], 4)
                    self.assertEqual(case.calls["log"], 1)

    def test_zero_time_or_base_output_keeps_safe_speedup_and_raw_token_counts(self):
        for draft_time, base_time in ((0.0, 0.0), (2.0, 0.0), (0.0, 4.0), (2.0, 4.0)):
            for base_tokens in (0, 6):
                with self.subTest(draft=draft_time, base=base_time, tokens=base_tokens):
                    case = _DatasetRun(
                        self.evaluator,
                        count=1,
                        paired=True,
                        throughput_warmup_samples=0,
                    )
                    case.costs = {"draft": draft_time, "base": base_time}
                    case.output_tokens["base"] = base_tokens
                    row, _ = case.run()
                    self.assertEqual(row["elapsed_s"], draft_time)
                    self.assertEqual(row["base_elapsed_s"], base_time)
                    self.assertEqual(row["base_total_output_tokens"], base_tokens)
                    draft_tps = 4 / draft_time if draft_time else 0
                    base_tps = base_tokens / base_time if base_time else 0.0
                    self.assertEqual(row["output_tokens_per_second"], draft_tps)
                    self.assertEqual(row["base_output_tokens_per_second"], base_tps)
                    self.assertEqual(
                        row["speedup_vs_base"],
                        draft_tps / base_tps if base_tps else 0.0,
                    )
                    self.assertEqual(
                        case.details("log")[-1][-4:],
                        (draft_tps, base_tps, row["speedup_vs_base"], 2.0),
                    )

    def test_generation_failures_propagate_without_recording_partial_pairs(self):
        for failure_type in (RuntimeError, KeyboardInterrupt):
            for fail_at in (("draft", 1), ("base", 1), ("draft", 2), ("base", 2)):
                with self.subTest(failure=failure_type, event=fail_at):
                    case = _DatasetRun(self.evaluator, count=1, paired=True)
                    case.fail_at = fail_at
                    case.failure = failure_type("injected dataset failure")
                    with self.assertRaises(failure_type) as caught:
                        case.run()
                    self.assertIs(caught.exception, case.failure)
                    self.assertEqual(case.names[-1], fail_at[0])
                    self.assertEqual(case.calls[fail_at[0]], fail_at[1])
                    self.assertEqual("seed" in case.names, fail_at[1] == 2)
                    self.assertNotIn("stats", case.names)
                    self.assertNotIn("artifact", case.names)
                    self.assertEqual(case.stats.elapsed_s, 0.0)

    def test_base_shard_aggregation_uses_max_time_and_retains_zero_time_guard(self):
        rows = [
            reporting.summary_row(
                "group/custom",
                1,
                reporting.EvalStats(elapsed_s=elapsed, total_output_tokens=tokens),
            )
            for elapsed, tokens in ((2.0, 4), (3.0, 5))
        ]
        for row, elapsed, tokens in zip(rows, (4.0, 5.0), (6, 7), strict=True):
            row.update(base_elapsed_s=elapsed, base_total_output_tokens=tokens)
        result = reporting.aggregate_rows("group/custom", rows)
        self.assertEqual(result["base_elapsed_s"], 5.0)
        self.assertEqual(result["base_total_output_tokens"], 13)
        self.assertEqual(result["base_output_tokens_per_second"], 13 / 5.0)
        self.assertEqual(result["speedup_vs_base"], 3.0 / (13 / 5.0))
        for row in rows:
            row["base_elapsed_s"] = 0.0
        result = reporting.aggregate_rows("group/custom", rows)
        self.assertEqual(result["base_total_output_tokens"], 0)
        self.assertEqual(result["base_output_tokens_per_second"], 0.0)
        self.assertEqual(result["speedup_vs_base"], 0.0)


if __name__ == "__main__":
    unittest.main()
