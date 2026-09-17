"""Opt-in DSV4 target integration; importing this package does not import torch."""

ARCHITECTURE = "SpeculatorsDeepseekV4ForCausalLM"
HS_FORMAT = "deepseek_v4_mean_hc_head"
KV_CACHE_COMPAT_ENV = "SPECULATORS_DSV4_HS_BRIDGE"


def register():
    """Register a separate vLLM architecture, without replacing Qwen/native V4."""
    import os  # noqa: PLC0415

    from vllm import ModelRegistry  # noqa: PLC0415

    ModelRegistry.register_model(
        ARCHITECTURE,
        "speculators_dsv4.ascend:SpeculatorsDeepseekV4ForCausalLM",
    )
    if os.environ.get(KV_CACHE_COMPAT_ENV) == "1":
        from speculators_dsv4.kv_cache import (  # noqa: PLC0415
            install_kv_cache_compatibility,
        )

        install_kv_cache_compatibility()
