"""Opt-in MMuse backbone configuration, separate from baseline DFlash."""

from typing import Literal

from pydantic import BaseModel, Field


class MMuseBackboneConfigMixin(BaseModel):
    """Fields for verifier fusion, query conditioning, convolution and selection."""

    dflash_context_residual: bool = Field(
        default=False,
        description=(
            "Inject the last verifier hidden state available before each anchor into "
            "all slots of its draft block through a zero-gated residual."
        ),
    )

    dflash_block_position_embedding: bool = Field(
        default=False,
        description=(
            "Add a learned block-relative position embedding to DFlash query slots. "
            "The embedding is zero-initialized."
        ),
    )

    dflash_gated_layer_fusion: bool = Field(
        default=False,
        description=(
            "Add normalized, per-token softmax-gated auxiliary-layer fusion as a "
            "zero-gated residual over the baseline concatenation projection."
        ),
    )

    dflash2_dynamic_conv: bool = Field(
        default=False,
        description=(
            "Wrap every draft Attention and MLP sublayer with DFlash2 grouped "
            "content-conditioned causal convolutions."
        ),
    )

    dflash2_conv_kernel_size: int = Field(
        default=2,
        gt=0,
        description="Number of causal taps in each DFlash2 grouped convolution.",
    )

    dflash2_conv_group_size: int = Field(
        default=16,
        gt=0,
        description="Hidden channels sharing each DFlash2 dynamic coefficient.",
    )

    dflash2_candidate_selector: bool = Field(
        default=False,
        description=(
            "Re-rank each position's Top-K draft candidates using the previous "
            "token and the DFlash hidden state."
        ),
    )

    dflash2_selector_rank: int = Field(
        default=256,
        gt=0,
        description="Low-rank transition width of the DFlash2 candidate selector.",
    )

    dflash2_selector_top_k: int = Field(
        default=16,
        gt=0,
        description="Number of unary LM-head candidates retained per draft slot.",
    )

    dflash2_selector_search_mode: Literal["greedy", "global"] = Field(
        default="greedy",
        description=(
            "Path search used by the DFlash2 candidate selector. 'greedy' follows "
            "the public DFlash2 predecessor walk; 'global' uses Viterbi search over "
            "locally normalized probabilities in the complete block-local Top-K "
            "transition lattice."
        ),
    )

    dflash2_selector_loss_weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight of the restricted-Top-K verifier-distillation loss used to "
            "train the DFlash2 selector."
        ),
    )
