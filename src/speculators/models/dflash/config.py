from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)

from speculators import SpeculatorModelConfig

__all__ = [
    "DFlashSpeculatorConfig",
]


@SpeculatorModelConfig.register("dflash")
class DFlashSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for DFlash speculator with vocabulary mapping.

    DFlash features vocabulary mapping between draft (64K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    """

    speculators_model_type: Literal["dflash"] = "dflash"
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlashSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    block_size: int = Field(
        default=8,
        description=(
            "Default size of the draft block predicted with a forward pass of the model"
        ),
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the DFlash auxiliary hidden state layers",
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for masking",
    )

    sliding_window_non_causal: bool = Field(
        default=False,
        description="Use non-causal (bidirectional) masking within draft blocks for "
        "sliding window attention layers. Full attention layers are always "
        "bidirectional.",
    )

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

    sample_from_anchor: bool = Field(
        default=False,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
        ),
    )

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = Qwen3Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size
