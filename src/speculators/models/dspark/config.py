from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus a Markov logit-bias head and a confidence head.

    The Markov head lets each draft position condition on previously sampled
    tokens within the block; the confidence head predicts the per-position
    acceptance probability. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["dspark"] = "dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
            "Default True matches DeepSeek/DeepSpec convention."
        ),
    )

    # Sequential (Markov) head.
    markov_rank: int = Field(
        default=256,
        description=(
            "Low-rank dimension of the Markov logit-bias factorization B = W1 @ W2. "
            "Set to 0 to disable the sequential head (pure DFlash drafting)."
        ),
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla",
        description=(
            "Sequential head variant: 'vanilla' (first-order Markov bias), 'gated' "
            "(hidden-gated bias), or 'rnn' (recurrent state over the block)."
        ),
    )

    # Causal correction head. When enabled it replaces the Markov head.
    enable_correction_head: bool = Field(
        default=False,
        description=(
            "Replace the Markov head with a tiny causal Transformer "
            "that predicts low-rank residual corrections to DFlash logits."
        ),
    )
    correction_hidden_size: int = Field(
        default=512,
        gt=0,
        description="Hidden width of the causal correction head.",
    )
    correction_rank: int = Field(
        default=256,
        gt=0,
        description="Low-rank bottleneck used to produce correction logits.",
    )
    correction_num_layers: int = Field(
        default=1,
        gt=0,
        description="Number of tiny causal Transformer layers.",
    )
    correction_num_heads: int = Field(
        default=8,
        gt=0,
        description="Number of attention heads in each correction layer.",
    )

    # Confidence head.
    enable_confidence_head: bool = Field(
        default=True,
        description="Whether to attach the per-position acceptance-probability head.",
    )
    confidence_head_with_markov: bool = Field(
        default=True,
        description=(
            "Concatenate the active sequential-head state (Markov embedding or "
            "causal correction state) with the backbone hidden state as the "
            "confidence-head input."
        ),
    )
