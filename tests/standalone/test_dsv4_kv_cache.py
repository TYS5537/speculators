"""Dependency-free tests for the pinned Ascend DSV4 hidden-state cache bridge."""

# ruff: noqa: PT009, PT027 -- Keep the suite runnable without pytest/torch.

import copy
import math
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_dsv4 import ARCHITECTURE, kv_cache
from speculators_dsv4.kv_cache import _CacheCompatibility


@dataclass(frozen=True)
class FakeMLASpec:
    block_size: int
    page_size_bytes: int
    compress_ratio: int = 1

    def max_memory_usage_bytes(self, config):
        return (
            math.ceil(config.model_config.max_model_len / self.block_size)
            * self.page_size_bytes
        )


@dataclass(frozen=True)
class FakeHiddenStateCacheSpec(FakeMLASpec):
    pass


@dataclass(frozen=True)
class FakeSlidingWindowSpec(FakeMLASpec):
    def max_memory_usage_bytes(self, config):
        tokens = min(config.model_config.max_model_len, 6)
        return math.ceil(tokens / self.block_size) * self.page_size_bytes


@dataclass(frozen=True)
class FakeUniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: dict

    @property
    def page_size_bytes(self):
        return sum(spec.page_size_bytes for spec in self.kv_cache_specs.values())

    def max_memory_usage_pages(self, config):
        return max(
            math.ceil(spec.max_memory_usage_bytes(config) / spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )

    def max_memory_usage_bytes(self, config):
        return self.max_memory_usage_pages(config) * self.page_size_bytes

    def get_page_sizes(self):
        return list({spec.page_size_bytes for spec in self.kv_cache_specs.values()})

    def get_num_layer_tuples(self):
        sizes = [spec.page_size_bytes for spec in self.kv_cache_specs.values()]
        return max(sizes.count(size) for size in sizes)


@dataclass
class FakeKVCacheGroupSpec:
    layer_names: list
    kv_cache_spec: object
    is_eagle_group: bool = False


@dataclass
class FakeKVCacheTensor:
    size: int
    shared_by: list
    offset: int = 0
    block_stride: int = 0


@dataclass
class FakeKVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list = field(default_factory=list)
    kv_cache_groups: list = field(default_factory=list)


def fake_config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=[ARCHITECTURE], model_type="deepseek_v4"
            ),
            max_model_len=16,
        ),
        speculative_config=SimpleNamespace(method="extract_hidden_states"),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )


class CacheCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.config = fake_config()
        self.specs = {
            "model.layers.0.attn": FakeMLASpec(8, 64, 4),
            "model.layers.0.indexer": FakeMLASpec(8, 32, 4),
            "model.layers.1.attn": FakeMLASpec(8, 64, 128),
            "model.layers.2.attn": FakeSlidingWindowSpec(2, 64),
            "draft.cache_only_layers.43": FakeHiddenStateCacheSpec(4, 512),
        }
        self.hs_name = "draft.cache_only_layers.43"
        self.interface = SimpleNamespace(
            HiddenStateCacheSpec=FakeHiddenStateCacheSpec,
            UniformTypeKVCacheSpecs=FakeUniformTypeKVCacheSpecs,
            KVCacheGroupSpec=FakeKVCacheGroupSpec,
            KVCacheTensor=FakeKVCacheTensor,
            KVCacheConfig=FakeKVCacheConfig,
        )
        names = (
            "get_kv_cache_groups",
            "get_kv_cache_config_from_groups",
            "_pool_bytes_per_block",
            "_max_memory_usage_bytes_from_groups",
            "get_max_concurrency_for_kv_cache_config",
        )
        self.originals = {name: Mock(name=name) for name in names}
        self.originals["get_kv_cache_groups"].side_effect = self.native_groups
        self.kv_utils = SimpleNamespace(**self.originals)
        self.native_allocate = Mock(side_effect=self.allocate_native)
        self.bridge = _CacheCompatibility(
            self.kv_utils, self.interface, self.native_allocate
        )
        self.bridge.install()
        self.native_cost = 96
        self.pool_cost = self.native_cost + 512

    def native_groups(self, _config, specs):
        """Reproduce the pinned planner's C4 / C128 / SWA assumption."""
        ratio_specs = {}
        sliding_specs = {}
        for name, spec in specs.items():
            if isinstance(spec, FakeSlidingWindowSpec):
                sliding_specs[name] = spec
            else:
                ratio_specs.setdefault(spec.compress_ratio, {})[name] = spec
        grouped_specs = [
            ratio_specs[ratio]
            for ratio in sorted(ratio_specs, key=lambda ratio: (ratio != 4, ratio))
        ]
        if sliding_specs:
            grouped_specs.append(sliding_specs)
        if any(
            not isinstance(spec, FakeSlidingWindowSpec)
            for group in grouped_specs[2:]
            for spec in group.values()
        ):
            raise AssertionError("Expected only SWA after the first two groups")
        return [
            FakeKVCacheGroupSpec(
                list(specs),
                FakeUniformTypeKVCacheSpecs(
                    next(iter(specs.values())).block_size, specs
                ),
            )
            for specs in grouped_specs
        ]

    def allocate_native(self, config, groups, available_memory):
        self.assertTrue(
            all(
                isinstance(group.kv_cache_spec, FakeUniformTypeKVCacheSpecs)
                for group in groups
            )
        )
        self.assertEqual(len(groups), 3)
        num_blocks = config.cache_config.num_gpu_blocks_override
        if num_blocks is None:
            num_blocks = available_memory // 96
        return num_blocks, [
            FakeKVCacheTensor(32 * num_blocks, [groups[0].layer_names[1]]),
            FakeKVCacheTensor(
                64 * num_blocks,
                [group.layer_names[0] for group in groups],
            ),
        ]

    def groups(self):
        return self.kv_utils.get_kv_cache_groups(self.config, self.specs)

    def allocate(self, num_blocks, extra=0):
        return self.kv_utils.get_kv_cache_config_from_groups(
            self.config, self.groups(), num_blocks * self.pool_cost + extra
        )

    def test_original_ratio_grouping_reproduces_hidden_state_failure(self):
        with self.assertRaisesRegex(AssertionError, "Expected only SWA"):
            self.native_groups(self.config, self.specs)

    def test_hidden_group_is_plain_and_native_specs_are_unchanged(self):
        before = copy.deepcopy(self.specs)
        groups = self.groups()
        self.assertEqual(len(groups), 4)
        self.assertEqual(self.specs, before)
        self.assertEqual(groups[-1].layer_names, [self.hs_name])
        self.assertIs(groups[-1].kv_cache_spec, self.specs[self.hs_name])
        self.assertFalse(groups[-1].is_eagle_group)
        self.assertEqual(
            [
                next(iter(group.kv_cache_spec.kv_cache_specs.values())).compress_ratio
                for group in groups[:-1]
            ],
            [4, 128, 1],
        )
        native_input = self.originals["get_kv_cache_groups"].call_args.args[1]
        self.assertNotIn(self.hs_name, native_input)
        self.assertEqual(set(native_input), set(self.specs) - {self.hs_name})

    def test_memory_budget_includes_native_and_dedicated_hidden_tensor(self):
        config = self.allocate(12, self.pool_cost - 1)
        self.assertEqual(config.num_blocks, 12)
        self.assertEqual(
            [tensor.size for tensor in config.kv_cache_tensors],
            [32 * 12, 64 * 12, 512 * 12],
        )
        self.assertEqual(
            sum(tensor.size for tensor in config.kv_cache_tensors),
            12 * self.pool_cost,
        )
        self.assertEqual(config.kv_cache_tensors[-1].shared_by, [self.hs_name])
        self.assertTrue(
            all(
                self.hs_name not in tensor.shared_by
                for tensor in config.kv_cache_tensors[:-1]
            )
        )
        self.assertEqual(
            self.kv_utils._pool_bytes_per_block(self.config, config.kv_cache_groups),
            self.pool_cost,
        )

    def test_native_unit_plan_uses_copy_without_mutating_caller_override(self):
        self.config.cache_config.num_gpu_blocks_override = 12
        before = copy.deepcopy(self.config)
        config = self.allocate(20)
        self.assertEqual(config.num_blocks, 12)
        self.assertEqual(self.config, before)
        for call in self.native_allocate.call_args_list:
            unit_config = call.args[0]
            self.assertIsNot(unit_config, self.config)
            self.assertIsNot(unit_config.cache_config, self.config.cache_config)
            self.assertEqual(unit_config.cache_config.num_gpu_blocks_override, 1)

    def test_global_worker_minimum_shrink_preserves_hidden_capacity(self):
        configs = [self.allocate(13), self.allocate(20)]
        minimum = min(config.num_blocks for config in configs)
        for config in configs:
            old_count = config.num_blocks
            config.num_blocks = minimum
            for tensor in config.kv_cache_tensors:
                self.assertEqual(tensor.size % old_count, 0)
                tensor.size = tensor.size // old_count * minimum
            self.assertEqual(config.kv_cache_tensors[-1].size, 512 * minimum)
            self.assertEqual(
                sum(tensor.size for tensor in config.kv_cache_tensors),
                self.pool_cost * minimum,
            )

    def test_full_request_memory_counts_all_groups_and_one_null_block(self):
        # C4=2, C128=2, SWA=3, HS=4. All consume distinct global IDs.
        required = self.kv_utils._max_memory_usage_bytes_from_groups(
            self.config, self.groups()
        )
        self.assertEqual(required, (2 + 2 + 3 + 4 + 1) * self.pool_cost)

    def test_connector_discovers_plain_hidden_group_on_worker_and_scheduler(self):
        config = self.allocate(12)
        for scheduler in (False, True):
            groups = copy.deepcopy(config.kv_cache_groups)
            if scheduler:
                for group in groups:
                    if isinstance(group.kv_cache_spec, FakeUniformTypeKVCacheSpecs):
                        group.kv_cache_spec = next(
                            iter(group.kv_cache_spec.kv_cache_specs.values())
                        )
            ids = [
                index
                for index, group in enumerate(groups)
                if isinstance(group.kv_cache_spec, FakeHiddenStateCacheSpec)
            ]
            self.assertEqual(ids, [3])
            self.assertEqual(groups[ids[0]].kv_cache_spec.block_size, 4)
            # request_finished_all_groups must forward the HS table, not C4.
            block_ids = ([1, 2], [3, 4], [5, 6, 7], [8, 9, 10, 11])
            self.assertEqual(block_ids[ids[0]], [8, 9, 10, 11])

    def test_concurrency_uses_global_pages_with_unwrapped_scheduler_groups(self):
        config = self.allocate(23)
        for scheduler in (False, True):
            with self.subTest(scheduler=scheduler):
                view = copy.deepcopy(config)
                if scheduler:
                    for group in view.kv_cache_groups:
                        spec = group.kv_cache_spec
                        if isinstance(spec, FakeUniformTypeKVCacheSpecs):
                            group.kv_cache_spec = next(
                                iter(spec.kv_cache_specs.values())
                            )
                self.assertEqual(
                    self.kv_utils.get_max_concurrency_for_kv_cache_config(
                        self.config, view
                    ),
                    2.0,
                )

    def test_unrelated_model_and_method_delegate_all_original_callbacks(self):
        configs = []
        for architecture, model_type, method in (
            ("Qwen3ForCausalLM", "qwen3", "extract_hidden_states"),
            ("DeepseekV4ForCausalLM", "deepseek_v4", "extract_hidden_states"),
            (ARCHITECTURE, "deepseek_v4", "dspark"),
        ):
            config = fake_config()
            config.model_config.hf_config.architectures = [architecture]
            config.model_config.hf_config.model_type = model_type
            config.speculative_config.method = method
            configs.append(config)
        no_speculation = fake_config()
        no_speculation.speculative_config = None
        configs.append(no_speculation)
        for config in configs:
            with self.subTest(config=config):
                for name, original in self.originals.items():
                    original.reset_mock()
                    original.side_effect = None
                    expected = object()
                    original.return_value = expected
                    argument = object()
                    args = (config, argument)
                    if name == "get_kv_cache_config_from_groups":
                        args += (12345,)
                    self.assertIs(getattr(self.kv_utils, name)(*args), expected)
                    original.assert_called_once_with(*args)

    def test_block_override_cannot_exceed_available_memory(self):
        self.config.cache_config.num_gpu_blocks_override = 13
        with self.assertRaises(ValueError):
            self.allocate(12)

    def test_allocation_without_one_global_block_fails_closed(self):
        with self.assertRaises(ValueError):
            self.allocate(0, self.pool_cost - 1)

    def test_allocation_that_cannot_admit_one_full_request_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.allocate(11)

    def test_nonuniform_native_worker_plan_fails_closed(self):
        groups = self.groups()
        groups[0].kv_cache_spec = next(
            iter(groups[0].kv_cache_spec.kv_cache_specs.values())
        )
        with self.assertRaises(ValueError):
            self.kv_utils.get_kv_cache_config_from_groups(
                self.config, groups, 12 * self.pool_cost
            )

    def test_wrapped_hidden_worker_group_fails_closed(self):
        groups = self.groups()
        groups[-1].kv_cache_spec = FakeUniformTypeKVCacheSpecs(
            4, {self.hs_name: self.specs[self.hs_name]}
        )
        with self.assertRaisesRegex(ValueError, "plain"):
            self.kv_utils.get_kv_cache_config_from_groups(
                self.config, groups, 12 * self.pool_cost
            )

    def test_multiple_hidden_caches_fail_closed(self):
        self.specs["draft.cache_only_layers.44"] = FakeHiddenStateCacheSpec(4, 512)
        with self.assertRaises(ValueError):
            self.groups()

    def test_installer_captures_platform_patches_and_is_idempotent(self):
        native_name = "vllm_ascend.patch.platform.patch_kv_cache_utils"
        utils_name = "vllm.v1.core.kv_cache_utils"
        interface_name = "vllm.v1.kv_cache_interface"
        fresh_utils = SimpleNamespace(**self.originals)
        platform_groups = Mock(name="platform_get_kv_cache_groups")
        native_module = SimpleNamespace(
            _get_kv_cache_config_deepseek_v4=self.native_allocate
        )
        imports = []

        def fake_import(name):
            imports.append(name)
            if name == native_name:
                # The actual platform import installs its own grouping wrapper.
                # Only its first import executes that module's body.
                if imports.count(name) == 1:
                    fresh_utils.get_kv_cache_groups = platform_groups
                return native_module
            if name == utils_name:
                self.assertIn(native_name, imports)
                return fresh_utils
            if name == interface_name:
                return self.interface
            raise AssertionError(f"Unexpected import: {name}")

        versions = {"vllm": "0.26.0+local", "vllm-ascend": "0.26.0rc1+local"}
        with (
            patch.object(kv_cache, "version", side_effect=versions.__getitem__),
            patch.object(kv_cache, "import_module", side_effect=fake_import),
        ):
            kv_cache.install_kv_cache_compatibility()
            self.assertEqual(imports, [native_name, utils_name, interface_name])
            installed = fresh_utils._speculators_dsv4_hs_cache
            self.assertIsInstance(installed, _CacheCompatibility)
            self.assertIs(installed.original["get_kv_cache_groups"], platform_groups)
            self.assertIs(installed.native_allocate, self.native_allocate)
            wrappers = {name: getattr(fresh_utils, name) for name in self.originals}
            for wrapper in wrappers.values():
                self.assertIs(wrapper.__self__, installed)

            kv_cache.install_kv_cache_compatibility()
            self.assertEqual(
                imports,
                [native_name, utils_name, interface_name, native_name, utils_name],
            )
            self.assertIs(fresh_utils._speculators_dsv4_hs_cache, installed)
            for name, wrapper in wrappers.items():
                self.assertIs(getattr(fresh_utils, name), wrapper)

    def test_installer_rejects_incompatible_versions_before_importing(self):
        for package, actual in (("vllm", "0.25.0"), ("vllm-ascend", "0.26.0rc2")):
            with self.subTest(package=package):
                versions = {"vllm": "0.26.0", "vllm-ascend": "0.26.0rc1"}
                versions[package] = actual
                with (
                    patch.object(kv_cache, "version", side_effect=versions.__getitem__),
                    patch.object(kv_cache, "import_module") as importer,
                    self.assertRaisesRegex(RuntimeError, package),
                ):
                    kv_cache.install_kv_cache_compatibility()
                importer.assert_not_called()


class CacheBindingTests(unittest.TestCase):
    def setUp(self):
        self.interface = SimpleNamespace(HiddenStateCacheSpec=FakeHiddenStateCacheSpec)
        self.config = fake_config()
        self.hs_name = "draft.cache_only_layers.43"
        self.native_name = "model.layers.0.attn"
        self.groups = [
            FakeKVCacheGroupSpec([self.native_name], FakeMLASpec(8, 64, 4)),
            FakeKVCacheGroupSpec([self.hs_name], FakeHiddenStateCacheSpec(4, 512)),
        ]
        self.cache_config = FakeKVCacheConfig(12, kv_cache_groups=self.groups)

        class FakeRunner:
            def __init__(self, config, caches):
                self.vllm_config = config
                self.caches = caches
                self.compilation_config = SimpleNamespace(
                    static_forward_context={
                        name: SimpleNamespace(kv_cache=None) for name in caches
                    }
                )
                self.seen_configs = []
                self.kv_caches = []

            def initialize_kv_cache_tensors(self, cache_config):
                self.seen_configs.append(cache_config)
                self.kv_caches = list(self.caches.values())
                for name, tensor in self.caches.items():
                    self.compilation_config.static_forward_context[name].kv_cache = [
                        tensor
                    ]
                return self.caches

        self.runner_class = FakeRunner
        self.caches = {self.native_name: object(), self.hs_name: object()}

    def test_bridge_binds_only_hidden_cache_as_bare_tensor(self):
        runner = self.runner_class(self.config, self.caches)
        kv_cache._install_cache_binding(self.runner_class, self.interface)
        result = runner.initialize_kv_cache_tensors(self.cache_config)
        layers = runner.compilation_config.static_forward_context
        self.assertIs(result, self.caches)
        self.assertIs(layers[self.hs_name].kv_cache, self.caches[self.hs_name])
        self.assertEqual(
            layers[self.native_name].kv_cache, [self.caches[self.native_name]]
        )
        self.assertEqual(runner.kv_caches, list(self.caches.values()))
        self.assertEqual(runner.seen_configs, [self.cache_config])

    def test_native_v4_qwen_and_nonextract_method_keep_original_bindings(self):
        kv_cache._install_cache_binding(self.runner_class, self.interface)
        for architecture, model_type, method in (
            ("DeepseekV4ForCausalLM", "deepseek_v4", "extract_hidden_states"),
            ("Qwen3ForCausalLM", "qwen3", "extract_hidden_states"),
            (ARCHITECTURE, "deepseek_v4", "dspark"),
        ):
            with self.subTest(architecture=architecture, method=method):
                config = fake_config()
                config.model_config.hf_config.architectures = [architecture]
                config.model_config.hf_config.model_type = model_type
                config.speculative_config.method = method
                runner = self.runner_class(config, self.caches)
                result = runner.initialize_kv_cache_tensors(self.cache_config)
                self.assertIs(result, self.caches)
                for (
                    name,
                    layer,
                ) in runner.compilation_config.static_forward_context.items():
                    self.assertEqual(layer.kv_cache, [self.caches[name]])
                self.assertEqual(runner.seen_configs, [self.cache_config])

    def test_binding_installation_does_not_wrap_twice(self):
        original = self.runner_class.initialize_kv_cache_tensors
        kv_cache._install_cache_binding(self.runner_class, self.interface)
        wrapper = self.runner_class.initialize_kv_cache_tensors
        kv_cache._install_cache_binding(self.runner_class, self.interface)
        self.assertIs(self.runner_class.initialize_kv_cache_tensors, wrapper)
        self.assertIs(wrapper.__wrapped__, original)
        runner = self.runner_class(self.config, self.caches)
        runner.initialize_kv_cache_tensors(self.cache_config)
        self.assertEqual(runner.seen_configs, [self.cache_config])

    def test_worker_installer_imports_runner_then_installs_binding(self):
        runner_name = "vllm_ascend.worker.model_runner_v1"
        interface_name = "vllm.v1.kv_cache_interface"
        modules = {
            runner_name: SimpleNamespace(NPUModelRunner=self.runner_class),
            interface_name: self.interface,
        }
        with (
            patch.object(
                kv_cache, "import_module", side_effect=modules.__getitem__
            ) as importer,
            patch.object(kv_cache, "_install_cache_binding") as installer,
        ):
            kv_cache.install_worker_cache_compatibility()
        self.assertEqual(
            [call.args for call in importer.call_args_list],
            [(runner_name,), (interface_name,)],
        )
        installer.assert_called_once_with(self.runner_class, self.interface)


if __name__ == "__main__":
    unittest.main()
