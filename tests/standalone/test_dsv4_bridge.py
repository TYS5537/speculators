"""Control-flow tests with a fake backend; these are NOT NPU integration tests."""

# ruff: noqa: PT009, PT027 -- Keep this suite runnable without pytest/torch.

import ast
import importlib.util
import io
import json
import shlex
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from speculators_dsv4 import ARCHITECTURE, register
from speculators_dsv4.block_protocol import BLOCK_CONNECTOR

ROOT = Path(__file__).resolve().parents[2]


def valid_config():
    return {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "hc_mult": 4,
        "vocab_size": 129280,
        "expert_dtype": "bf16",
        "torch_dtype": "bfloat16",
    }


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.ndim = 2
        self.shape = (4, 4096)

    def detach(self):
        return self

    def clone(self):
        return FakeTensor(self.value)


class FakeNativeModel:
    def __init__(self, *, vllm_config, prefix):
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model = SimpleNamespace(
            norm=SimpleNamespace(
                register_forward_pre_hook=lambda hook: setattr(self, "hook", hook)
            )
        )

    def set_aux_hidden_state_layers(self, layers):
        self.layers = layers

    def forward(self, *args):
        teacher = FakeTensor(7)
        self.hook(None, (teacher,))
        teacher.value = 99  # Simulate an in-place norm backend.
        self.normalized = FakeTensor(2)
        self.auxiliary = [FakeTensor(i) for i in self.layers]
        return self.normalized, self.auxiliary


def runtime_config():
    config = valid_config()
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(**config, to_dict=lambda: dict(config)),
            enforce_eager=True,
            dtype="bf16",
        ),
        quant_config=None,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
        speculative_config=SimpleNamespace(method="extract_hidden_states"),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
    )


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch = ModuleType("torch")
        torch.bfloat16 = "bf16"
        native = ModuleType("vllm_ascend.models.deepseek_v4")
        native.AscendDeepseekV4ForCausalLM = FakeNativeModel
        ascend_config = ModuleType("vllm_ascend.ascend_config")
        self.ascend = SimpleNamespace(enable_flashcomm1=False, enable_dsa_cp=False)
        ascend_config.get_ascend_config = lambda: self.ascend
        with patch.dict(
            sys.modules,
            {
                "torch": torch,
                "vllm_ascend.models.deepseek_v4": native,
                "vllm_ascend.ascend_config": ascend_config,
            },
        ):
            self.module = load_module(
                "dsv4_test_runtime", ROOT / "src/speculators_dsv4/ascend.py"
            )
        self.version = patch.object(
            self.module,
            "version",
            side_effect={
                "vllm": "0.26.0",
                "vllm-ascend": "0.26.0rc1+test",
            }.__getitem__,
        )
        self.version.start()
        self.addCleanup(self.version.stop)

    def make_model(self, config=None):
        return self.module.SpeculatorsDeepseekV4ForCausalLM(
            vllm_config=config or runtime_config()
        )

    def test_teacher_capture_does_not_mutate_target_or_auxiliary(self):
        model = self.make_model()
        model.set_aux_hidden_state_layers((1, 11, 21, 30, 40, 43))
        normalized, auxiliary = model.forward(None, None)
        self.assertIs(normalized, model.normalized)
        for i in range(5):
            self.assertIs(auxiliary[i], model.auxiliary[i])
        self.assertEqual(auxiliary[-1].value, 7)
        self.assertEqual(model.auxiliary[-1].value, 43)
        self.assertIsNone(model._teacher_pre_norm)

    def test_block_export_hook_receives_corrected_teacher_only_when_enabled(self):
        connector = ModuleType("speculators_dsv4.block_connector")
        connector.export_block = Mock()
        input_ids, positions = object(), object()
        with patch.dict(sys.modules, {connector.__name__: None}):
            reference = self.make_model()
            reference.set_aux_hidden_state_layers((1, 11, 21, 30, 40, 43))
            reference.forward(input_ids, positions)
            connector.export_block.assert_not_called()
        with patch.dict(sys.modules, {connector.__name__: connector}):
            config = runtime_config()
            config.kv_transfer_config = SimpleNamespace(kv_connector=BLOCK_CONNECTOR)
            model = self.make_model(config)
            model.set_aux_hidden_state_layers((1, 11, 21, 30, 40, 43))
            output = model.forward(input_ids, positions)
        connector.export_block.assert_called_once_with(
            model, input_ids, positions, output
        )
        self.assertIs(connector.export_block.call_args.args[3], output)
        self.assertEqual(output[1][-1].value, 7)
        self.assertEqual(model.auxiliary[-1].value, 43)
        self.assertIsNone(model._teacher_pre_norm)

    def test_quantized_target_config_is_passed_to_native_backend(self):
        config = runtime_config()
        config.quant_config = object()
        metadata = {**valid_config(), "expert_dtype": "fp4"}
        metadata["quantization_config"] = {"quant_method": "fp8"}
        config.model_config.hf_config = SimpleNamespace(
            **metadata, to_dict=lambda: dict(metadata)
        )
        model = self.make_model(config)
        self.assertIs(model.vllm_config, config)
        self.assertIs(model.vllm_config.quant_config, config.quant_config)
        self.assertEqual(model.vllm_config.model_config.dtype, "bf16")
        model.set_aux_hidden_state_layers((1, 11, 21, 30, 40, 43))
        normalized, auxiliary = model.forward(None, None)
        self.assertIs(normalized, model.normalized)
        self.assertEqual(auxiliary[-1].value, 7)

    def test_quantized_target_still_requires_bf16_runtime(self):
        config = runtime_config()
        config.quant_config = object()
        config.model_config.dtype = "fp16"
        with self.assertRaisesRegex(ValueError, "dtype bfloat16"):
            self.make_model(config)

    def test_explicit_layer_setup_is_required(self):
        model = self.make_model()
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            model.forward(None, None)
        for layers in ((1, 40), (43,), (0, 43), (11, 1, 43)):
            with self.assertRaises(ValueError):
                model.set_aux_hidden_state_layers(layers)

    def test_rejects_unsupported_runtime_modes(self):
        for section, field, value in (
            ("model_config", "enforce_eager", False),
            ("model_config", "dtype", "fp16"),
            ("cache_config", "enable_prefix_caching", True),
            ("scheduler_config", "enable_chunked_prefill", True),
            ("speculative_config", "method", "dspark"),
            ("parallel_config", "pipeline_parallel_size", 2),
            ("parallel_config", "data_parallel_size", 2),
            ("parallel_config", "prefill_context_parallel_size", 2),
            ("parallel_config", "decode_context_parallel_size", 2),
        ):
            config = runtime_config()
            setattr(getattr(config, section), field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.make_model(config)
        config = runtime_config()
        config.compilation_config.pass_config.enable_sp = True
        with self.assertRaises(ValueError):
            self.make_model(config)
        self.ascend.enable_flashcomm1 = True
        with self.assertRaises(ValueError):
            self.make_model()

    def test_rejects_other_runtime_versions(self):
        with (
            patch.object(self.module, "version", return_value="0.25.0"),
            self.assertRaisesRegex(RuntimeError, "requires"),
        ):
            self.make_model()

    def test_plugin_only_registers_new_architecture_lazily(self):
        fake_vllm = ModuleType("vllm")
        fake_vllm.ModelRegistry = SimpleNamespace(register_model=Mock())
        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            register()
        fake_vllm.ModelRegistry.register_model.assert_called_once_with(
            ARCHITECTURE, "speculators_dsv4.ascend:SpeculatorsDeepseekV4ForCausalLM"
        )


class LauncherTests(unittest.TestCase):
    def setUp(self):
        # Force the dependency-free inline file connector for these CLI tests.
        with patch.dict(sys.modules, {"hs_connectors": None}):
            self.launcher = load_module(
                "dsv4_test_launcher", ROOT / "scripts/launch_vllm.py"
            )

    def launch_dsv4(self, flag="--dsv4", extra=(), *, block=False):
        argv = [
            "launch_vllm.py",
            "fixture",
            flag,
            "--hidden-states-path",
            "fixture-hs",
            *(["--dsv4-block-verify"] if block else []),
            "--",
            *extra,
        ]
        report = {
            "config": {**valid_config(), "expert_dtype": "fp4"},
            "model_path": "fixture",
            "checkpoint_signature": "fixture",
        }
        with (
            patch.object(sys, "argv", argv),
            patch(
                "speculators_dsv4.contract.inspect_checkpoint", return_value=report
            ) as inspect,
            patch("speculators_dsv4.contract.ensure_manifest") as manifest,
            patch(
                "importlib.metadata.entry_points",
                return_value=[SimpleNamespace(name="speculators_dsv4")],
            ),
            patch.dict(
                self.launcher.os.environ, {"VLLM_PLUGINS": "ascend,speculators_dsv4"}
            ),
            patch.object(self.launcher.os, "execvp") as execute,
            redirect_stdout(io.StringIO()),
        ):
            self.launcher.main()
        inspect.assert_called_once_with("fixture")
        self.manifest_call = manifest.call_args
        return execute.call_args.args[1]

    def test_dsv4_primary_flag_and_legacy_alias_are_equivalent(self):
        self.assertEqual(self.launch_dsv4(), self.launch_dsv4("--dsv4-bf16"))

    def test_dsv4_passes_quantization_options_unchanged(self):
        for extra, method in (
            (["--quantization", "fp8"], "fp8"),
            (["--quantization=ascend"], "ascend"),
            (["-q", "ascend"], "ascend"),
        ):
            for flag in ("--dsv4", "--dsv4-bf16"):
                with self.subTest(extra=extra, flag=flag):
                    cmd = self.launch_dsv4(flag, extra)
                    start = cmd.index(extra[0])
                    self.assertEqual(cmd[start : start + len(extra)], extra)
                    self.assertEqual(cmd[cmd.index("--dtype") + 1], "bfloat16")
                    self.assertEqual(
                        self.manifest_call.kwargs["runtime_quantization"],
                        {"method": method},
                    )

    def test_dsv4_records_backend_auto_quantization_mode(self):
        self.launch_dsv4()
        self.assertEqual(
            self.manifest_call.kwargs["runtime_quantization"], {"method": None}
        )

    def test_block_verification_selects_dedicated_connector_and_single_sequence(self):
        cmd = self.launch_dsv4(block=True)
        connector = json.loads(cmd[cmd.index("--kv_transfer_config") + 1])
        self.assertEqual(
            connector,
            {
                "kv_connector": "DSV4BlockVerifyConnector",
                "kv_connector_module_path": "speculators_dsv4.block_connector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {"shared_storage_path": "fixture-hs"},
            },
        )
        speculative = json.loads(cmd[cmd.index("--speculative_config") + 1])
        self.assertEqual(speculative["method"], "extract_hidden_states")
        self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "1")

    def test_reference_and_training_keep_original_hidden_state_connector(self):
        cmd = self.launch_dsv4()
        connector = json.loads(cmd[cmd.index("--kv_transfer_config") + 1])
        self.assertEqual(connector["kv_connector"], "ExampleHiddenStatesConnector")
        self.assertNotIn("kv_connector_module_path", connector)
        self.assertNotIn("--max-num-seqs", cmd)

    def test_block_verification_rejects_multiple_sequence_configuration(self):
        for extra in (
            ["--max-num-seqs", "2"],
            ["--max-num-seqs=2"],
            ["--max_num_seqs", "2"],
            ["--max-num-seqs", "2", "--max-num-seqs", "1"],
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(ValueError, "max-num-seqs 1"),
            ):
                self.launch_dsv4(extra=extra, block=True)

    def test_block_verification_requires_explicit_dsv4_bridge(self):
        with (
            patch.object(
                sys, "argv", ["launch_vllm.py", "fixture", "--dsv4-block-verify"]
            ),
            self.assertRaisesRegex(ValueError, "requires --dsv4"),
        ):
            self.launcher.main()

    def test_dsv4_quantized_dry_run_does_not_write_manifest_or_launch(self):
        report = {
            "config": valid_config(),
            "model_path": "fixture",
            "checkpoint_signature": "fixture",
        }
        with (
            patch.object(
                sys,
                "argv",
                [
                    "launch_vllm.py",
                    "fixture",
                    "--dsv4",
                    "--dry-run",
                    "--",
                    "-q",
                    "ascend",
                ],
            ),
            patch("speculators_dsv4.contract.inspect_checkpoint", return_value=report),
            patch("speculators_dsv4.contract.ensure_manifest") as manifest,
            patch("importlib.metadata.entry_points", return_value=[]),
            patch.dict(self.launcher.os.environ, {"VLLM_PLUGINS": "speculators_dsv4"}),
            patch.object(self.launcher.os, "execvp") as execute,
            redirect_stdout(io.StringIO()) as output,
        ):
            self.launcher.main()
        manifest.assert_not_called()
        execute.assert_not_called()
        self.assertIn("-q ascend", output.getvalue())

    def test_dsv4_still_rejects_dtype_hf_and_hidden_state_overrides(self):
        for option in (
            "--dtype",
            "--hf-overrides",
            "--speculative-config",
            "--speculative_config",
            "--kv-transfer-config",
            "--kv_transfer_config",
        ):
            for extra in ([option, "override"], [f"{option}=override"]):
                with (
                    self.subTest(extra=extra),
                    self.assertRaisesRegex(ValueError, "owns"),
                ):
                    self.launch_dsv4(extra=extra)

    def test_dsv4_command_and_manifest_teacher_semantics(self):
        argv = [
            "launch_vllm.py",
            "fixture",
            "--dsv4-bf16",
            "--hidden-states-path",
            "fixture-hs",
            "--",
            "--tensor-parallel-size",
            "8",
        ]
        report = {
            "config": valid_config(),
            "model_path": "fixture",
            "checkpoint_signature": "fixture",
        }
        with (
            patch.object(sys, "argv", argv),
            patch("speculators_dsv4.contract.inspect_checkpoint", return_value=report),
            patch("speculators_dsv4.contract.ensure_manifest") as manifest,
            patch(
                "importlib.metadata.entry_points",
                return_value=[SimpleNamespace(name="speculators_dsv4")],
            ),
            patch.dict(
                self.launcher.os.environ, {"VLLM_PLUGINS": "ascend,speculators_dsv4"}
            ),
            patch.object(self.launcher.os, "execvp") as execute,
            redirect_stdout(io.StringIO()),
        ):
            self.launcher.main()
        cmd = execute.call_args.args[1]
        speculative = json.loads(cmd[cmd.index("--speculative_config") + 1])
        ids = speculative["draft_model_config"]["hf_config"][
            "eagle_aux_hidden_state_layer_ids"
        ]
        self.assertEqual(ids, [1, 11, 21, 30, 40, 43])
        self.assertEqual(
            manifest.call_args.args[1]["auxiliary_hs_ids"], [1, 11, 21, 30, 40]
        )
        self.assertIn("--enforce-eager", cmd)
        self.assertIn("--no-enable-prefix-caching", cmd)
        self.assertIn("--no-enable-chunked-prefill", cmd)

    def test_standard_qwen_command_stays_on_original_path(self):
        fake_hf = ModuleType("transformers")
        fake_hf.AutoConfig = SimpleNamespace(
            from_pretrained=lambda _: SimpleNamespace(
                num_hidden_layers=36, model_type="qwen3"
            )
        )
        with (
            patch.dict(sys.modules, {"transformers": fake_hf}),
            patch.object(sys, "argv", ["launch_vllm.py", "qwen-fixture"]),
            patch.object(self.launcher.os, "execvp") as execute,
            redirect_stdout(io.StringIO()),
        ):
            self.launcher.main()
        cmd = execute.call_args.args[1]
        speculative = json.loads(cmd[cmd.index("--speculative_config") + 1])
        self.assertEqual(
            speculative["draft_model_config"]["hf_config"][
                "eagle_aux_hidden_state_layer_ids"
            ],
            [2, 18, 33, 36],
        )
        self.assertNotIn("--hf-overrides", cmd)
        self.assertNotIn("--dtype", cmd)


class CheckpointCliTests(unittest.TestCase):
    def test_general_and_strict_audits_have_separate_entry_points(self):
        for script, kwargs in (
            ("check_dsv4_checkpoint.py", {}),
            ("check_dsv4_bf16.py", {"require_bf16": True}),
        ):
            with self.subTest(script=script):
                module = load_module(
                    f"test_{script.removesuffix('.py')}", ROOT / "scripts" / script
                )
                report = {"config": valid_config(), "weight_dtypes": {"BF16": 3}}
                with (
                    patch.object(sys, "argv", [script, "fixture"]),
                    patch.object(
                        module, "inspect_checkpoint", return_value=report
                    ) as inspect,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    module.main()
                inspect.assert_called_once_with("fixture", **kwargs)
                self.assertIn("NOT validated", output.getvalue())


class RecipeTests(unittest.TestCase):
    def test_trainer_flags_are_real_and_pin_user_recipe(self):
        path = ROOT / "examples/train/dspark_dsv4_flash_bf16_trainer.sh"
        source = path.read_text(encoding="utf-8")
        self.assertIn("TRAIN_ENTRY=(scripts/train.py)", source)
        invocation = source.split('"${TRAIN_ENTRY[@]}" \\\n', 1)[1]
        args = shlex.split(invocation.replace("\\\n", " "), comments=True)
        for name, value in {
            "--scheduler-type": "linear",
            "--optimizer": "muon",
            "--lr": "6e-5",
            "--correction-gate-bias": "0",
            "--correction-markov-gate-bias": "-2.0",
            "--epochs": "5",
            "--block-size": "7",
            "--total-seq-len": "3072",
        }.items():
            self.assertEqual(args[args.index(name) + 1], value)
        tree = ast.parse((ROOT / "scripts/train.py").read_text(encoding="utf-8"))
        declared = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("--")
        }
        # The HS file backend adds this flag outside the training script.
        declared.add("--hidden-states-path")
        for arg in args:
            if arg.startswith("--no-"):
                arg = "--" + arg.removeprefix("--no-")
            if arg.startswith("--"):
                self.assertIn(arg, declared)

    def test_dense_geometry_and_linux_line_endings(self):
        config = json.loads(
            (ROOT / "examples/train/dsv4_flash_dense_config.json").read_text()
        )
        self.assertEqual(config["model_type"], "qwen3")
        for key, value in {
            "hidden_size": 4096,
            "vocab_size": 129280,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 9728,
            "sliding_window": 2048,
        }.items():
            self.assertEqual(config[key], value)
        self.assertEqual(config["layer_types"], ["sliding_attention"] * 5)
        for script in (ROOT / "examples/train").glob("dspark_dsv4_flash_bf16_*.sh"):
            self.assertNotIn(b"\r\n", script.read_bytes())

    def test_offline_eval_requires_explicit_v4_service_backend(self):
        source = ROOT / "scripts/evaluate/dspark_offline_eval.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_target_cache_support"
        )
        scope = {}
        exec(  # noqa: S102 -- Compile only the named helper in this checkout.
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            scope,
        )
        fake_hf = ModuleType("transformers")
        config = {"model_type": "deepseek_v4"}
        fake_hf.PretrainedConfig = SimpleNamespace(
            get_config_dict=lambda _: (config, {})
        )
        with patch.dict(sys.modules, {"transformers": fake_hf}):
            with self.assertRaisesRegex(NotImplementedError, "DynamicCache.crop"):
                scope["_validate_target_cache_support"]("fixture")
            self.assertEqual(
                scope["_validate_target_cache_support"]("fixture", "dsv4-vllm"),
                config,
            )
            config["model_type"] = "qwen3"
            self.assertEqual(scope["_validate_target_cache_support"]("fixture"), config)
            with self.assertRaises(ValueError):
                scope["_validate_target_cache_support"]("fixture", "dsv4-vllm")


if __name__ == "__main__":
    unittest.main()
