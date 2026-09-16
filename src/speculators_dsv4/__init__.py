"""Opt-in DSV4 target integration; importing this package does not import torch."""

ARCHITECTURE = "SpeculatorsDeepseekV4ForCausalLM"
HS_FORMAT = "deepseek_v4_mean_hc_head"


def register():
    """Register a separate vLLM architecture, without replacing Qwen/native V4."""
    from vllm import ModelRegistry  # noqa: PLC0415

    ModelRegistry.register_model(
        ARCHITECTURE,
        "speculators_dsv4.ascend:SpeculatorsDeepseekV4ForCausalLM",
    )
