"""Scoped KV planning for Ascend V4 plus the cache-only HS extraction layer.

Keep Ascend's compressed-attention packing, but give HS its own physical tensor
and scheduler group. All tensors still use the same global block-ID capacity.
This module deliberately has no torch/vLLM imports until installation.
"""

# ruff: noqa: SLF001 -- Pinned, opt-in compatibility with backend-private APIs.

import copy
from functools import wraps
from importlib import import_module
from importlib.metadata import version

from speculators_dsv4 import ARCHITECTURE

_METHODS = (
    "get_kv_cache_groups",
    "get_kv_cache_config_from_groups",
    "_pool_bytes_per_block",
    "_max_memory_usage_bytes_from_groups",
    "get_max_concurrency_for_kv_cache_config",
)


def _is_bridge(vllm_config):
    model = getattr(vllm_config, "model_config", None)
    hf = getattr(model, "hf_config", None)
    return (
        getattr(hf, "architectures", None) == [ARCHITECTURE]
        and getattr(hf, "model_type", None) == "deepseek_v4"
        and getattr(getattr(vllm_config, "speculative_config", None), "method", None)
        == "extract_hidden_states"
    )


class _CacheCompatibility:
    def __init__(self, kv_utils, interface, native_allocate):
        self.utils = kv_utils
        self.types = interface
        self.native_allocate = native_allocate
        self.original = {name: getattr(kv_utils, name) for name in _METHODS}

    def install(self):
        for name in _METHODS:
            setattr(self.utils, name, getattr(self, name))

    def _split_groups(self, groups):
        hidden = [
            group
            for group in groups
            if isinstance(group.kv_cache_spec, self.types.HiddenStateCacheSpec)
        ]
        native = [group for group in groups if group not in hidden]
        if len(hidden) != 1 or len(hidden[0].layer_names) != 1:
            raise ValueError(
                "DSV4 HS cache requires exactly one plain hidden-state group."
            )
        if not native or any(
            not isinstance(group.kv_cache_spec, self.types.UniformTypeKVCacheSpecs)
            or any(
                isinstance(spec, self.types.HiddenStateCacheSpec)
                for spec in group.kv_cache_spec.kv_cache_specs.values()
            )
            for group in native
        ):
            raise ValueError(
                "DSV4 target cache groups must use native UniformType packing."
            )
        return native, hidden[0]

    def get_kv_cache_groups(self, vllm_config, kv_cache_spec):
        if not _is_bridge(vllm_config):
            return self.original["get_kv_cache_groups"](vllm_config, kv_cache_spec)
        # HiddenStateCacheSpec subclasses MLA; remove it BEFORE Ascend groups MLA
        # by compression ratio (otherwise its ratio=1 displaces the C128 group).
        hidden = {
            name: spec
            for name, spec in kv_cache_spec.items()
            if isinstance(spec, self.types.HiddenStateCacheSpec)
        }
        if len(hidden) != 1:
            raise ValueError("DSV4 HS cache requires exactly one hidden-state layer.")
        native = {
            name: spec for name, spec in kv_cache_spec.items() if name not in hidden
        }
        groups = list(self.original["get_kv_cache_groups"](vllm_config, native))
        name, spec = next(iter(hidden.items()))
        # Keep the plain type: ExampleHiddenStatesConnector discovers it directly.
        # Do not realign its page/block size to a compressed-attention page.
        groups.append(self.types.KVCacheGroupSpec([name], spec))
        self._split_groups(groups)
        return groups

    def _unit_plan(self, vllm_config, kv_cache_groups):
        native, hidden = self._split_groups(kv_cache_groups)
        # Ask the pinned native allocator for metadata for one global block. This
        # preserves all C4/C128/SWA sharing and padding without copying its logic.
        planning_config = copy.copy(vllm_config)
        planning_config.cache_config = copy.copy(vllm_config.cache_config)
        planning_config.cache_config.num_gpu_blocks_override = 1
        num_blocks, tensors = self.native_allocate(planning_config, native, 0)
        if num_blocks != 1:
            raise ValueError("DSV4 native allocator did not produce a one-block plan.")
        tensors = [
            *tensors,
            self.types.KVCacheTensor(
                size=hidden.kv_cache_spec.page_size_bytes,
                shared_by=list(hidden.layer_names),
            ),
        ]
        expected = sorted(
            name for group in kv_cache_groups for name in group.layer_names
        )
        actual = sorted(name for tensor in tensors for name in tensor.shared_by)
        if actual != expected or any(tensor.size <= 0 for tensor in tensors):
            raise ValueError(
                "DSV4 cache allocation must cover each layer exactly once."
            )
        return tensors, sum(tensor.size for tensor in tensors)

    def _required_pages(self, vllm_config, groups):
        pages = 0
        for group in groups:
            spec = group.kv_cache_spec
            if isinstance(spec, self.types.UniformTypeKVCacheSpecs):
                pages += spec.max_memory_usage_pages(vllm_config)
            else:
                # The scheduler unwraps native UniformType groups to single specs.
                size = spec.page_size_bytes
                pages += (spec.max_memory_usage_bytes(vllm_config) + size - 1) // size
        return pages

    def get_kv_cache_config_from_groups(
        self, vllm_config, kv_cache_groups, available_memory
    ):
        if not _is_bridge(vllm_config):
            return self.original["get_kv_cache_config_from_groups"](
                vllm_config, kv_cache_groups, available_memory
            )
        unit_tensors, cost = self._unit_plan(vllm_config, kv_cache_groups)
        override = vllm_config.cache_config.num_gpu_blocks_override
        num_blocks = available_memory // cost if override is None else override
        if num_blocks <= 0 or num_blocks * cost > available_memory:
            raise ValueError(
                "Insufficient memory for the requested DSV4 HS cache blocks."
            )
        # All groups draw distinct IDs from one pool, with one global null block.
        needed = self._required_pages(vllm_config, kv_cache_groups) + 1
        if num_blocks < needed:
            raise ValueError(
                f"Insufficient DSV4 HS cache capacity: {num_blocks} blocks available, "
                f"{needed} required (including the null block). Reduce max-model-len "
                "or increase the target cache memory budget."
            )
        return self.types.KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                self.types.KVCacheTensor(
                    size=tensor.size * num_blocks, shared_by=list(tensor.shared_by)
                )
                for tensor in unit_tensors
            ],
            kv_cache_groups=kv_cache_groups,
        )

    def _pool_bytes_per_block(self, vllm_config, kv_cache_groups):
        if not _is_bridge(vllm_config):
            return self.original["_pool_bytes_per_block"](vllm_config, kv_cache_groups)
        return self._unit_plan(vllm_config, kv_cache_groups)[1]

    def _max_memory_usage_bytes_from_groups(self, vllm_config, kv_cache_groups):
        if not _is_bridge(vllm_config):
            return self.original["_max_memory_usage_bytes_from_groups"](
                vllm_config, kv_cache_groups
            )
        cost = self._unit_plan(vllm_config, kv_cache_groups)[1]
        return (self._required_pages(vllm_config, kv_cache_groups) + 1) * cost

    def get_max_concurrency_for_kv_cache_config(self, vllm_config, kv_cache_config):
        if not _is_bridge(vllm_config):
            return self.original["get_max_concurrency_for_kv_cache_config"](
                vllm_config, kv_cache_config
            )
        pages = self._required_pages(vllm_config, kv_cache_config.kv_cache_groups)
        return max(0, kv_cache_config.num_blocks - 1) / pages if pages else 0.0


def install_kv_cache_compatibility():
    """Install before EngineCore plans caches, only in launcher-opted-in processes."""
    for package, expected in (("vllm", "0.26.0"), ("vllm-ascend", "0.26.0rc1")):
        actual = version(package).split("+")[0]
        if actual != expected:
            raise RuntimeError(
                f"DSV4 HS cache bridge requires {package}=={expected}, got {actual}"
            )
    # General plugins load before Ascend's platform patches. Import those first
    # so a subsequent normal platform import cannot overwrite our wrappers.
    native = import_module("vllm_ascend.patch.platform.patch_kv_cache_utils")
    utils = import_module("vllm.v1.core.kv_cache_utils")
    if getattr(utils, "_speculators_dsv4_hs_cache", None) is not None:
        return
    interface = import_module("vllm.v1.kv_cache_interface")
    bridge = _CacheCompatibility(
        utils, interface, native._get_kv_cache_config_deepseek_v4
    )
    bridge.install()
    utils._speculators_dsv4_hs_cache = bridge


def _install_cache_binding(runner_class, interface):
    original = runner_class.initialize_kv_cache_tensors
    if getattr(original, "_speculators_dsv4_hs_binding", False):
        return

    @wraps(original)
    def initialize_kv_cache_tensors(runner, kv_cache_config):
        caches = original(runner, kv_cache_config)
        if _is_bridge(runner.vllm_config):
            for group in kv_cache_config.kv_cache_groups:
                if isinstance(group.kv_cache_spec, interface.HiddenStateCacheSpec):
                    for name in group.layer_names:
                        # Ascend V4 binds [tensor] for its attention layers, but
                        # upstream CacheOnlyAttentionLayer consumes a bare tensor.
                        layer = runner.compilation_config.static_forward_context[name]
                        layer.kv_cache = caches[name]
        return caches

    initialize_kv_cache_tensors._speculators_dsv4_hs_binding = True
    runner_class.initialize_kv_cache_tensors = initialize_kv_cache_tensors


def install_worker_cache_compatibility():
    """Called during model construction, after the worker runner module is loaded."""
    runner = import_module("vllm_ascend.worker.model_runner_v1")
    interface = import_module("vllm.v1.kv_cache_interface")
    _install_cache_binding(runner.NPUModelRunner, interface)
