"""Evaluator setup/dispatch/resource contracts with no model or service imports."""

# ruff: noqa: PT009, PT027 -- Also runs with stdlib unittest.

import builtins
import importlib.util
import itertools
import os
import sys
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts/evaluate/dspark_offline_eval.py"


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


class _Model:
    def __init__(self, case, name, dtype):
        self.case = case
        self.name = name
        self.dtype = dtype
        self.config = case.config
        self.block_size = 3
        self.correction_head = None
        self.markov_head = None

    def to(self, *args, **kwargs):
        self.case.record(f"{self.name}_to", *args, **kwargs)
        return self

    def eval(self):
        self.case.record(f"{self.name}_eval")
        return self


class _Case:
    def __init__(self, args):
        self.args = args
        self.events = []
        self.fail_at = None
        self.failure = RuntimeError("injected failure")
        self.clients = []
        self.config = SimpleNamespace(
            speculators_model_type="mmuse",
            sample_from_anchor=True,
            transformer_layer_config=SimpleNamespace(_attn_implementation="saved"),
            correction_output_mode="logits",
            markov_head_type="gated",
        )
        self.report = {
            "model_path": "local-target",
            "config": {"vocab_size": 32},
        }
        self.target = _Model(self, "target", "resolved-target-dtype")
        self.remote = SimpleNamespace(
            model_name="resolved-served-model",
            generation_config=SimpleNamespace(eos_token_id=[2, 1]),
        )
        self.draft = _Model(self, "draft", "checkpoint-dtype")
        self.tokenizer = object()
        self.runner, self.base_runner = object(), object()
        self.d2t, self.t2d = object(), object()
        self.stop_ids = [2, 1]

    @property
    def names(self):
        return [name for name, _, _ in self.events]

    def record(self, name, *args, **kwargs):
        self.events.append((name, args, kwargs))
        if name == self.fail_at:
            raise self.failure

    def stub(self, name, result=None):
        def call(*args, **kwargs):
            self.record(name, *args, **kwargs)
            return result

        return Mock(side_effect=call)

    def event(self, name):
        return next(
            (args, kwargs) for event, args, kwargs in self.events if event == name
        )

    def open_client(self, **kwargs):
        index = len(self.clients)
        self.record(f"client{index}", **kwargs)
        client = SimpleNamespace(close=self.stub(f"close{index}"))
        self.clients.append(client)
        return client

    def evaluate(self, **kwargs):
        dataset = kwargs["path"].stem
        self.record("evaluate:" + dataset, **kwargs)
        return {"dataset": dataset}, [{"sample": dataset}]

    def install(self, evaluator, stack):
        patches = {
            "_validate_target_cache_support": self.stub(
                "validate", self.report["config"]
            ),
            "_write_backend_metadata": self.stub("metadata"),
            "_load_draft_config": self.stub("draft_config", self.config),
            "_load_vocab_mapping_tensors": self.stub(
                "vocab_files", (self.d2t, self.t2d)
            ),
            "_ensure_loaded_vocab_mappings": self.stub("vocab_check"),
            "DSparkOfflineRunner": self.stub("runner", self.runner),
            "BaseModelOfflineRunner": self.stub("base_runner", self.base_runner),
            "resolve_stop_token_ids": self.stub("stop", self.stop_ids),
            "_discover_datasets": self.stub(
                "datasets", [Path("alpha.jsonl"), Path("beta.jsonl")]
            ),
            "_evaluate_dataset": self.evaluate,
            "_write_outputs": self.stub("outputs"),
            "run_ascend_data_parallel": self.stub("dispatch"),
            "logger": Mock(),
            "torch": None,
            "DynamicCache": None,
        }
        for name, value in patches.items():
            stack.enter_context(patch.object(evaluator, name, value))
        modules = {
            "torch": _module(
                "torch",
                manual_seed=self.stub("seed"),
                device=str,
                bfloat16="bfloat16",
                float32="float32",
            ),
            "transformers": _module(
                "transformers",
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=self.stub("hf_tokenizer", self.tokenizer)
                ),
                AutoModelForCausalLM=SimpleNamespace(
                    from_pretrained=self.stub("hf_model", self.target)
                ),
                DynamicCache=object(),
            ),
            "speculators": _module("speculators", __path__=[]),
            "speculators.model": _module(
                "speculators.model",
                SpeculatorModel=SimpleNamespace(
                    from_pretrained=self.stub("draft_load", self.draft)
                ),
            ),
            "openai": _module("openai", OpenAI=self.open_client),
            "speculators_dsv4": _module("speculators_dsv4", __path__=[]),
        }
        for name, attributes in {
            "contract": {"inspect_checkpoint": self.stub("inspect", self.report)},
            "eval_contract": {"bind_draft_verifier": self.stub("bind")},
            "offline": {"DSV4OfflineTarget": self.stub("remote_target", self.remote)},
            "tokenizer": {
                "DSV4ServerTokenizer": self.stub("tokenizer", self.tokenizer)
            },
            "hs_http": {
                "validate_endpoint": self.stub(
                    "http_endpoint", "http://hs.invalid/normalized"
                ),
                "validate_token": self.stub("http_token"),
            },
        }.items():
            full_name = "speculators_dsv4." + name
            modules[full_name] = _module(full_name, **attributes)
        stack.enter_context(patch.dict(sys.modules, modules))
        stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "test-api-key",
                    "DSV4_HS_HTTP_TOKEN": "test-hs-token",
                },
            )
        )


class OfflineEvalLifecycleTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "eval_lifecycle_fixture", EVALUATOR
        )
        self.evaluator = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.evaluator
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(self.evaluator)

    @contextmanager
    def case(self, backend="hf", **options):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    [
                        "eval",
                        "--verifier-model",
                        "local-target",
                        "--draft-model",
                        "local-draft",
                        "--datasets-root",
                        "datasets",
                        "--target-backend",
                        backend,
                        "--device",
                        "cpu",
                        "--datasets",
                        "alpha,beta",
                        "--vllm-endpoint",
                        "http://target.invalid/prefix/v1/",
                        "--hidden-states-path",
                        "hidden-states",
                    ],
                )
            )
            args = self.evaluator.parse_args()
            vars(args).update(options)
            case = _Case(args)
            case.install(self.evaluator, stack)
            yield case

    def assert_dataset_contract(self, case):
        for dataset in ("alpha", "beta"):
            _, kwargs = case.event("evaluate:" + dataset)
            self.assertIs(kwargs["args"], case.args)
            self.assertIs(kwargs["runner"], case.runner)
            self.assertIs(kwargs["stop_token_ids"], case.stop_ids)
            self.assertIs(
                kwargs["base_runner"],
                case.base_runner if case.args.measure_base_speedup else None,
            )
        args, _ = case.event("outputs")
        self.assertEqual(args[0], case.args.output_dir)
        self.assertEqual(args[1], [{"dataset": "alpha"}, {"dataset": "beta"}])
        self.assertEqual(
            args[2],
            {}
            if case.args.skip_artifacts
            else {
                "alpha": [{"sample": "alpha"}],
                "beta": [{"sample": "beta"}],
            },
        )

    def test_hf_order_precision_and_output_contracts(self):
        for dtype, measure, skip in itertools.product(
            ("auto", "bfloat16", "float32"), (False, True), (False, True)
        ):
            with (
                self.subTest(dtype=dtype, measure=measure, skip=skip),
                self.case(
                    dtype=dtype,
                    measure_base_speedup=measure,
                    skip_artifacts=skip,
                    temperature=0.7,
                ) as case,
            ):
                self.evaluator.run(case.args)
                self.assertEqual(
                    case.names,
                    [
                        "validate",
                        "seed",
                        "hf_tokenizer",
                        "hf_model",
                        "target_to",
                        "target_eval",
                        "draft_config",
                        "vocab_files",
                        "draft_load",
                        "draft_to",
                        "draft_eval",
                        "vocab_check",
                        "runner",
                        *(["base_runner"] if measure else []),
                        "stop",
                        "datasets",
                        "evaluate:alpha",
                        "evaluate:beta",
                        "outputs",
                    ],
                )
                self.assertEqual(case.event("hf_model")[1]["torch_dtype"], dtype)
                self.assertEqual(
                    case.event("draft_load")[1]["torch_dtype"], case.target.dtype
                )
                self.assertEqual(
                    case.event("draft_to")[1],
                    {"device": "cpu", "dtype": case.target.dtype},
                )
                self.assertIs(case.event("draft_load")[1]["d2t"], case.d2t)
                self.assertIs(case.event("draft_load")[1]["t2d"], case.t2d)
                self.assertEqual(self.evaluator.logger.warning.call_count, int(measure))
                self.assert_dataset_contract(case)

    def test_dsv4_order_transport_arguments_and_client_lifetime(self):
        for mode, http in itertools.product(("reference", "block"), (False, True)):
            with (
                self.subTest(mode=mode, http=http),
                self.case(
                    "dsv4-vllm",
                    dsv4_verification_mode=mode,
                    hs_http_endpoint="http://hs.invalid/" if http else None,
                    hidden_states_path=None if http else Path("hidden-states"),
                ) as case,
            ):
                self.evaluator.run(case.args)
                self.assertEqual(
                    case.names,
                    [
                        *(["http_endpoint", "http_token"] if http else []),
                        "validate",
                        "inspect",
                        "metadata",
                        "seed",
                        "draft_config",
                        "bind",
                        "vocab_files",
                        "draft_load",
                        "draft_to",
                        "draft_eval",
                        "vocab_check",
                        "client0",
                        "remote_target",
                        "client1",
                        "tokenizer",
                        "runner",
                        "stop",
                        "datasets",
                        "evaluate:alpha",
                        "evaluate:beta",
                        "outputs",
                        "close1",
                        "close0",
                    ],
                )
                self.assertEqual(case.event("draft_load")[1]["torch_dtype"], "bfloat16")
                client_options = {
                    "api_key": "test-api-key",
                    "timeout": case.args.target_request_timeout,
                    "max_retries": 0,
                }
                self.assertEqual(
                    case.event("client0")[1],
                    {**client_options, "base_url": case.args.vllm_endpoint},
                )
                self.assertEqual(
                    case.event("client1")[1],
                    {**client_options, "base_url": "http://target.invalid/prefix"},
                )
                target_args, target_options = case.event("remote_target")
                self.assertEqual(target_args, (case.draft, case.report))
                self.assertIs(target_options["client"], case.clients[0])
                self.assertEqual(target_options["verification_mode"], mode)
                self.assertEqual(
                    target_options["hs_http_endpoint"],
                    "http://hs.invalid/normalized" if http else None,
                )
                self.assertEqual(target_options["hs_http_token"], "test-hs-token")
                self.assertEqual(
                    target_options["hidden_states_path"], case.args.hidden_states_path
                )
                self.assertEqual(
                    case.event("tokenizer"),
                    (
                        (case.clients[1], "resolved-served-model", [2, 1]),
                        {"vocab_size": 32},
                    ),
                )
                self.assert_dataset_contract(case)

    def test_parent_dispatch_finishes_preflight_without_loading_model_backends(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "transformers", "speculators", "openai"}:
                raise AssertionError("parent imported model backend: " + name)
            return original_import(name, *args, **kwargs)

        for backend in ("hf", "dsv4-vllm"):
            with (
                self.subTest(backend=backend),
                self.case(backend, ascend_devices="0,1") as case,
            ):
                with patch.object(builtins, "__import__", guarded_import):
                    self.evaluator.run(case.args)
                self.assertEqual(
                    case.names,
                    [
                        "validate",
                        *(["inspect", "metadata"] if backend == "dsv4-vllm" else []),
                        "dispatch",
                    ],
                )
                self.assertIsNone(self.evaluator.torch)
                self.assertIsNone(self.evaluator.DynamicCache)

    def test_worker_with_parent_device_list_does_not_dispatch_again(self):
        with self.case(ascend_devices="0,1", worker_shard_index=1) as case:
            self.evaluator.run(case.args)
            self.assertNotIn("dispatch", case.names)
            self.assertIn("outputs", case.names)

    def test_invalid_remote_options_fail_before_inspection_or_loading(self):
        for options, message in (
            ({"vllm_endpoint": None}, "DSV4 needs"),
            ({"hidden_states_path": None}, "DSV4 needs"),
            ({"measure_base_speedup": True}, "not online base speedup"),
            ({"dtype": "auto"}, "requires --dtype bfloat16"),
        ):
            with (
                self.subTest(options=options),
                self.case("dsv4-vllm", **options) as case,
            ):
                with self.assertRaisesRegex(ValueError, message):
                    self.evaluator.run(case.args)
                self.assertEqual(case.names, ["validate"])
        with self.case(hs_http_endpoint="http://hs.invalid/") as case:
            with self.assertRaisesRegex(ValueError, "requires --target-backend"):
                self.evaluator.run(case.args)
            self.assertEqual(case.names, [])

    def test_failure_and_interrupt_close_exactly_the_acquired_clients(self):
        stages = (
            "inspect",
            "metadata",
            "draft_config",
            "bind",
            "vocab_files",
            "draft_load",
            "draft_to",
            "vocab_check",
            "client0",
            "remote_target",
            "client1",
            "tokenizer",
            "runner",
            "stop",
            "datasets",
            "evaluate:alpha",
            "evaluate:beta",
            "outputs",
        )
        for stage, error_type in itertools.product(
            stages, (RuntimeError, KeyboardInterrupt)
        ):
            with (
                self.subTest(stage=stage, error=error_type),
                self.case("dsv4-vllm") as case,
            ):
                case.fail_at, case.failure = stage, error_type("injected failure")
                with self.assertRaises(error_type) as raised:
                    self.evaluator.run(case.args)
                self.assertIs(raised.exception, case.failure)
                closed = [name for name in case.names if name.startswith("close")]
                self.assertEqual(
                    closed, [f"close{i}" for i in reversed(range(len(case.clients)))]
                )
                for client in case.clients:
                    client.close.assert_called_once_with()
                self.assertEqual(case.names[case.names.index(stage) + 1 :], closed)

    def test_close_failure_still_closes_earlier_client(self):
        with self.case("dsv4-vllm") as case:
            case.fail_at = "close1"
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                self.evaluator.run(case.args)
            self.assertEqual(case.names[-2:], ["close1", "close0"])

    def test_anchor_attention_and_sequential_head_contracts(self):
        for correction, markov, anchor in itertools.product(
            (False, True), (False, True), (None, "true", "false")
        ):
            with (
                self.subTest(correction=correction, markov=markov, anchor=anchor),
                self.case(
                    sample_from_anchor=anchor,
                    device="npu:0",
                ) as case,
            ):
                case.draft.correction_head = (
                    SimpleNamespace(position_embedding=object()) if correction else None
                )
                case.draft.markov_head = object() if markov else None
                self.evaluator.run(case.args)
                self.assertEqual(case.config.sample_from_anchor, anchor != "false")
                self.assertEqual(
                    case.config.transformer_layer_config._attn_implementation, "sdpa"
                )
                log_args = self.evaluator.logger.info.call_args_list[0].args
                expected = "correction:logits" if correction else ""
                if markov:
                    expected += ("+" if correction else "") + "markov:gated"
                self.assertEqual(log_args[5], expected or "none")
        with self.case("dsv4-vllm") as case:
            case.draft.correction_head = object()
            with self.assertRaisesRegex(RuntimeError, "native causal CorrectionHead"):
                self.evaluator.run(case.args)
            self.assertNotIn("runner", case.names)
            self.assertEqual(case.names[-2:], ["close1", "close0"])


if __name__ == "__main__":
    unittest.main()
