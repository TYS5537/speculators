"""Dependency-free regression tests for the scoped Ascend target graph check."""

# ruff: noqa: PT009, PT027 -- Keep the suite runnable without pytest/torch.

import copy
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

from speculators_dsv4 import ARCHITECTURE, graph


class FakeGraphMode(Enum):
    NONE = 0
    FULL_DECODE_ONLY = 1
    FULL = 2


class FakeHiddenStateCacheSpec:
    pass


class FakeTargetBackend:
    support = "UNIFORM_BATCH"


class FakeCacheOnlyBackend:
    support = "NEVER"


class FakeUnsupportedTargetBackend:
    support = "NEVER"


def fake_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[ARCHITECTURE], model_type="deepseek_v4"
            ),
            enforce_eager=False,
        ),
        speculative_config=SimpleNamespace(method="extract_hidden_states"),
        compilation_config=SimpleNamespace(
            mode=0, cudagraph_mode=FakeGraphMode.FULL_DECODE_ONLY
        ),
    )


class GraphCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.interface = SimpleNamespace(HiddenStateCacheSpec=FakeHiddenStateCacheSpec)
        self.groups = [
            SimpleNamespace(layer_names=["target.c4"], kv_cache_spec=object()),
            SimpleNamespace(
                layer_names=["draft.cache_only_layers.43"],
                kv_cache_spec=FakeHiddenStateCacheSpec(),
            ),
            SimpleNamespace(layer_names=["target.swa"], kv_cache_spec=object()),
        ]
        self.backends = [
            {FakeTargetBackend},
            {FakeCacheOnlyBackend},
            {FakeTargetBackend},
        ]

        class FakeRunner:
            def __init__(self, config):
                self.vllm_config = config
                self.compilation_config = config.compilation_config
                self.calls = []
                self.checked = []
                self.result = object()
                self.native_error = None
                self.forced_mode = None

            def _check_and_update_cudagraph_mode(self, backends, groups):
                self.calls.append((backends, groups))
                if self.native_error is not None:
                    raise self.native_error
                for backend_set in backends:
                    for backend in backend_set:
                        self.checked.append(backend)
                        if backend.support == "NEVER":
                            self.compilation_config.cudagraph_mode = FakeGraphMode.NONE
                if self.forced_mode is not None:
                    self.compilation_config.cudagraph_mode = self.forced_mode
                return self.result

        self.runner_class = FakeRunner
        self.runner = FakeRunner(fake_config())

    def install(self):
        graph._install_graph_compatibility(
            self.runner_class, self.interface, FakeCacheOnlyBackend
        )

    def check(self, runner=None):
        return (runner or self.runner)._check_and_update_cudagraph_mode(
            self.backends, self.groups
        )

    def assert_original_inputs_unchanged(self):
        self.assertEqual(
            self.backends,
            [{FakeTargetBackend}, {FakeCacheOnlyBackend}, {FakeTargetBackend}],
        )
        self.assertEqual(FakeCacheOnlyBackend.support, "NEVER")
        self.assertIsInstance(self.groups[1].kv_cache_spec, FakeHiddenStateCacheSpec)

    def test_unpatched_cache_only_capability_disables_target_graph(self):
        self.check()
        self.assertEqual(
            self.runner.compilation_config.cudagraph_mode, FakeGraphMode.NONE
        )

    def test_only_capability_input_is_filtered_preserving_group_indices(self):
        # The instance already exists when the model constructor installs the
        # wrapper; later KV initialization must still resolve the new method.
        self.install()
        result = self.check()
        self.assertIs(result, self.runner.result)
        passed_backends, passed_groups = self.runner.calls[0]
        self.assertIsNot(passed_backends, self.backends)
        self.assertIs(passed_groups, self.groups)
        self.assertIs(passed_backends[0], self.backends[0])
        self.assertEqual(passed_backends[1], set())
        self.assertIs(passed_backends[2], self.backends[2])
        self.assertEqual(self.runner.checked, [FakeTargetBackend, FakeTargetBackend])
        self.assertEqual(
            self.runner.compilation_config.cudagraph_mode,
            FakeGraphMode.FULL_DECODE_ONLY,
        )
        self.assert_original_inputs_unchanged()

    def test_native_backend_capability_is_not_overridden(self):
        self.install()
        self.backends[2] = {FakeUnsupportedTargetBackend}
        with self.assertRaisesRegex(RuntimeError, "native backend.*NONE"):
            self.check()
        self.assertIn(FakeUnsupportedTargetBackend, self.runner.checked)
        self.assertEqual(self.backends[1], {FakeCacheOnlyBackend})

    def test_any_native_graph_mode_change_fails_explicitly(self):
        self.install()
        self.runner.forced_mode = FakeGraphMode.FULL
        with self.assertRaisesRegex(RuntimeError, "disabled or changed"):
            self.check()
        self.assert_original_inputs_unchanged()

    def test_native_exception_propagates_without_mutating_groups(self):
        self.install()
        error = ValueError("native attention failure")
        self.runner.native_error = error
        with self.assertRaises(ValueError) as caught:
            self.check()
        self.assertIs(caught.exception, error)
        self.assert_original_inputs_unchanged()

    def test_other_architectures_methods_and_execution_modes_delegate_exactly(self):
        self.install()
        variations = (
            ("architecture", "DeepseekV4ForCausalLM"),
            ("architecture", "Qwen3ForCausalLM"),
            ("model_type", "qwen3"),
            ("method", "dspark"),
            ("graph", FakeGraphMode.NONE),
            ("graph", FakeGraphMode.FULL),
            ("compile", 3),
            ("eager", True),
        )
        for field, value in variations:
            with self.subTest(field=field, value=value):
                config = fake_config()
                if field == "architecture":
                    config.model_config.hf_config.architectures = [value]
                elif field == "model_type":
                    config.model_config.hf_config.model_type = value
                elif field == "method":
                    config.speculative_config.method = value
                elif field == "graph":
                    config.compilation_config.cudagraph_mode = value
                elif field == "compile":
                    config.compilation_config.mode = value
                else:
                    config.model_config.enforce_eager = value
                runner = self.runner_class(config)
                self.assertIs(self.check(runner), runner.result)
                self.assertIs(runner.calls[0][0], self.backends)
                self.assertIs(runner.calls[0][1], self.groups)
                self.assertIn(FakeCacheOnlyBackend, runner.checked)

    def test_unexpected_hidden_group_layout_is_rejected_before_native_call(self):
        self.install()
        for case in ("misaligned", "missing", "duplicate", "layers", "backend"):
            with self.subTest(case=case):
                groups = copy.deepcopy(self.groups)
                backends = [set(value) for value in self.backends]
                if case == "misaligned":
                    backends.pop()
                elif case == "missing":
                    groups[1].kv_cache_spec = object()
                elif case == "duplicate":
                    groups[2].kv_cache_spec = FakeHiddenStateCacheSpec()
                elif case == "layers":
                    groups[1].layer_names.append("unexpected")
                else:
                    backends[1].add(FakeTargetBackend)
                with self.assertRaises(ValueError):
                    self.runner._check_and_update_cudagraph_mode(backends, groups)
                self.assertEqual(self.runner.calls, [])

    def test_hidden_only_topology_cannot_claim_target_graph_support(self):
        self.install()
        with self.assertRaisesRegex(ValueError, "native target attention"):
            self.runner._check_and_update_cudagraph_mode(
                [self.backends[1]], [self.groups[1]]
            )

    def test_install_is_idempotent(self):
        original = self.runner_class._check_and_update_cudagraph_mode
        self.install()
        wrapper = self.runner_class._check_and_update_cudagraph_mode
        self.install()
        self.assertIs(self.runner_class._check_and_update_cudagraph_mode, wrapper)
        self.assertIs(wrapper.__wrapped__, original)
        self.check()
        self.assertEqual(len(self.runner.calls), 1)

    def test_missing_pinned_runner_interface_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "pinned Ascend V1"):
            graph._install_graph_compatibility(
                type("IncompatibleRunner", (), {}), self.interface, FakeCacheOnlyBackend
            )

    def test_worker_installer_defers_imports_and_installs_once(self):
        modules = {
            "vllm_ascend.worker.model_runner_v1": SimpleNamespace(
                NPUModelRunner=self.runner_class
            ),
            "vllm.v1.kv_cache_interface": self.interface,
            "vllm.model_executor.models.extract_hidden_states": SimpleNamespace(
                CacheOnlyAttentionBackend=FakeCacheOnlyBackend
            ),
        }
        with patch.object(
            graph, "import_module", side_effect=modules.__getitem__
        ) as imp:
            graph.install_worker_graph_compatibility()
            self.assertEqual(
                [call.args[0] for call in imp.call_args_list], list(modules)
            )
            wrapper = self.runner_class._check_and_update_cudagraph_mode
            graph.install_worker_graph_compatibility()
            self.assertIs(self.runner_class._check_and_update_cudagraph_mode, wrapper)
        self.check()
        self.assertEqual(len(self.runner.calls), 1)


if __name__ == "__main__":
    unittest.main()
