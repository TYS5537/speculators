"""Scoped target graph-capability checks for the pinned Ascend HS bridge.

The cache-only layer runs separately in the extractor proposer, not inside the
target's ACL graph. Its eager-only capability must not disable the target graph.
No torch/vLLM modules are imported until worker installation.
"""

# ruff: noqa: SLF001 -- Pinned, opt-in compatibility with backend-private APIs.

from functools import wraps
from importlib import import_module

from speculators_dsv4.kv_cache import _is_bridge


def _is_target_graph(vllm_config):
    compilation = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    return (
        _is_bridge(vllm_config)
        and not vllm_config.model_config.enforce_eager
        and getattr(compilation, "mode", None) == 0
        and getattr(mode, "name", mode) == "FULL_DECODE_ONLY"
    )


def _install_graph_compatibility(runner_class, interface, cache_only_backend):
    original = getattr(runner_class, "_check_and_update_cudagraph_mode", None)
    if not callable(original):
        raise RuntimeError("DSV4 HS graph bridge requires the pinned Ascend V1 runner.")
    if getattr(original, "_speculators_dsv4_hs_graph", False):
        return

    @wraps(original)
    def check_and_update(runner, attention_backends, kv_cache_groups):
        if not _is_target_graph(runner.vllm_config):
            return original(runner, attention_backends, kv_cache_groups)
        if len(attention_backends) != len(kv_cache_groups):
            raise ValueError("DSV4 HS graph backend and cache groups must align.")
        hidden = [
            index
            for index, group in enumerate(kv_cache_groups)
            if isinstance(group.kv_cache_spec, interface.HiddenStateCacheSpec)
        ]
        if len(hidden) != 1 or len(kv_cache_groups[hidden[0]].layer_names) != 1:
            raise ValueError("DSV4 HS graph requires one plain hidden-state group.")
        index = hidden[0]
        if attention_backends[index] != {cache_only_backend}:
            raise ValueError("DSV4 HS graph requires the native cache-only backend.")
        if not any(
            backends for i, backends in enumerate(attention_backends) if i != index
        ):
            raise ValueError("DSV4 HS graph requires native target attention backends.")

        # Only this capability-check input is filtered. Keep group indices and
        # every native backend intact; do not touch runner state, the original
        # collections, metadata builders, or CacheOnly's declared capability.
        # FULL_DECODE_ONLY has mixed_mode=NONE, so the extractor's separate
        # dispatcher remains eager in the pinned upstream implementation.
        target_backends = list(attention_backends)
        target_backends[index] = set()
        result = original(runner, target_backends, kv_cache_groups)
        resolved = runner.compilation_config.cudagraph_mode
        if getattr(resolved, "name", resolved) != "FULL_DECODE_ONLY":
            raise RuntimeError(
                "DSV4 HS target FULL_DECODE_ONLY graph was disabled or changed by "
                f"the native backend (resolved {getattr(resolved, 'name', resolved)}). "
                "Check backend compatibility or explicitly select eager execution."
            )
        return result

    check_and_update._speculators_dsv4_hs_graph = True
    runner_class._check_and_update_cudagraph_mode = check_and_update


def install_worker_graph_compatibility():
    """Call during model construction after the version-checked native init.

    At this point the V1 runner module is loaded. Its later KV-cache initialization
    calls initialize_attn_backend, which invokes the patched capability check
    before constructing metadata builders. Installing during a general plugin
    import would instead risk a worker/model import cycle.
    """
    runner = import_module("vllm_ascend.worker.model_runner_v1")
    interface = import_module("vllm.v1.kv_cache_interface")
    extractor = import_module("vllm.model_executor.models.extract_hidden_states")
    _install_graph_compatibility(
        runner.NPUModelRunner, interface, extractor.CacheOnlyAttentionBackend
    )
