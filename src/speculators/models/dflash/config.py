from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)

from speculators import SpeculatorModelConfig

__all__ = [
    "DFlashSpeculatorConfig",
    "GlmMoeDsaDraftConfig",
]


class GlmMoeDsaDraftConfig(PretrainedConfig):
    """Draft decoder config for GLM/DeepSeek-style MLA + MoE DFlash layers."""

    model_type = "glm_moe_dsa"

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 6144,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 1,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        q_lora_rank: int | None = 2048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 8,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 2048,
        first_k_dense_replace: int = 3,
        scoring_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        index_topk: int | None = 2048,
        index_topk_freq: int = 4,
        index_skip_topk_offset: int = 3,
        index_topk_pattern: list[str] | None = None,
        indexer_rope_interleave: bool = True,
        rope_interleave: bool = True,
        rope_parameters: dict[str, Any] | None = None,
        sliding_window: int | None = None,
        layer_types: list[str] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.index_topk = index_topk
        self.index_topk_freq = index_topk_freq
        self.index_skip_topk_offset = index_skip_topk_offset
        self.index_topk_pattern = index_topk_pattern
        self.indexer_rope_interleave = indexer_rope_interleave
        self.rope_interleave = rope_interleave
        self.rope_parameters = rope_parameters or {
            "rope_type": "default",
            "rope_theta": 10000.0,
        }
        self.sliding_window = sliding_window
        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers


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

    max_anchors: int = Field(
        default=256,
        description=(
            "Maximum number of anchor positions to sample during training "
            "(controls memory usage and training efficiency)"
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
                if value["model_type"] == GlmMoeDsaDraftConfig.model_type:
                    config_class = GlmMoeDsaDraftConfig
                else:
                    config_class = AutoConfig.for_model(
                        model_type=value["model_type"]
                    ).__class__
            return config_class(**value)
        return value

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size
