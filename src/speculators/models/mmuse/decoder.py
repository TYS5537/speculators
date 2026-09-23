"""MMuse decoder extensions over the unchanged DFlash attention/MLP layer."""

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs, Qwen3Config
from typing_extensions import Unpack

from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.mmuse.dynamic_conv import DFlash2GroupedConv


class Qwen3MMuseDecoderLayer(Qwen3DFlashDecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        *,
        dflash2_dynamic_conv: bool = False,
        dflash2_conv_kernel_size: int = 2,
        dflash2_conv_group_size: int = 16,
        block_size: int = 8,
    ):
        super().__init__(config=config, layer_idx=layer_idx)
        self.attention_conv: DFlash2GroupedConv | None = None
        self.mlp_conv: DFlash2GroupedConv | None = None
        if dflash2_dynamic_conv:
            conv_kwargs = {
                "hidden_size": config.hidden_size,
                "taps": dflash2_conv_kernel_size,
                "group_size": dflash2_conv_group_size,
                "block_size": block_size,
            }
            self.attention_conv = DFlash2GroupedConv(**conv_kwargs)
            self.mlp_conv = DFlash2GroupedConv(**conv_kwargs)

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        # necessary, but kept here for BC
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # The main difference between this method and the qwen 3 layer it is
        # built from is that it
        # passes the extra hidden states to the self attention from the verifier model.
        # Note that target_hidden is not modified here.
        assert hidden_states is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_coefficients = None
        if self.attention_conv is not None:
            hidden_states, attention_coefficients = self.attention_conv.prepare(
                hidden_states
            )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        if self.attention_conv is not None:
            assert attention_coefficients is not None  # noqa: S101
            hidden_states = self.attention_conv.finish(
                hidden_states, attention_coefficients
            )
        hidden_states = residual + hidden_states  # type: ignore[operator]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_coefficients = None
        if self.mlp_conv is not None:
            hidden_states, mlp_coefficients = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.mlp_conv is not None:
            assert mlp_coefficients is not None  # noqa: S101
            hidden_states = self.mlp_conv.finish(hidden_states, mlp_coefficients)
        return residual + hidden_states  # type: ignore[operator,return-value]
