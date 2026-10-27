"""KV-conditioned decoder with explicit, immutable draft-cache inputs.

Only token steps append cache entries. A caller owns one cache per logical
depth (loop, physical layer), even though all loops share the layer weights.
"""

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward,
)

from speculators.models.dflash.model_definitions import (
    Qwen3DFlashAttention,
    apply_rotary_pos_emb,
)

DraftKV = tuple[torch.Tensor, torch.Tensor]


class LEagleAttention(Qwen3DFlashAttention):
    def project_context(self, target_hidden, position_embeddings) -> DraftKV:
        """Project a verified context suffix once for inference cache reuse."""
        batch, length = target_hidden.shape[:2]
        key = self.k_norm(
            self.k_proj(target_hidden).view(batch, length, -1, self.head_dim)
        ).transpose(1, 2)
        _, key = apply_rotary_pos_emb(key, key, *position_embeddings)
        value = (
            self.v_proj(target_hidden)
            .view(batch, length, -1, self.head_dim)
            .transpose(1, 2)
        )
        return key, value

    def forward(
        self,
        hidden_states,
        target_hidden,
        position_embeddings,
        attention_mask,
        past_draft_kv: DraftKV | None = None,
        context_kv: DraftKV | None = None,
    ) -> tuple[torch.Tensor, DraftKV]:
        batch, query_length = hidden_states.shape[:2]
        context_length = target_hidden.shape[1]
        query = self.q_norm(
            self.q_proj(hidden_states).view(batch, query_length, -1, self.head_dim)
        ).transpose(1, 2)
        if context_kv is None:
            key = torch.cat(
                [self.k_proj(target_hidden), self.k_proj(hidden_states)], dim=1
            ).view(batch, context_length + query_length, -1, self.head_dim)
            key = self.k_norm(key).transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
            context_key, draft_key = key.split([context_length, query_length], dim=-2)
            context_value = (
                self.v_proj(target_hidden)
                .view(batch, context_length, -1, self.head_dim)
                .transpose(1, 2)
            )
        else:
            if self.training or torch.is_grad_enabled():
                raise ValueError("Precomputed LEAGLE context KV is inference-only")
            context_key, context_value = context_kv
            draft_key = self.k_norm(
                self.k_proj(hidden_states).view(batch, query_length, -1, self.head_dim)
            ).transpose(1, 2)
            query, draft_key = apply_rotary_pos_emb(
                query, draft_key, *position_embeddings
            )
        draft_value = (
            self.v_proj(hidden_states)
            .view(batch, query_length, -1, self.head_dim)
            .transpose(1, 2)
        )
        if past_draft_kv is not None:
            draft_key = torch.cat([past_draft_kv[0], draft_key], dim=-2)
            draft_value = torch.cat([past_draft_kv[1], draft_value], dim=-2)
        # The immutable target prefix is never appended to the draft cache.
        key = torch.cat([context_key, draft_key], dim=-2)
        value = torch.cat([context_value, draft_value], dim=-2)
        implementation = self.config._attn_implementation  # noqa: SLF001
        attention = (
            eager_attention_forward
            if implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[implementation]
        )
        output, _ = attention(
            self,
            query,
            key,
            value,
            attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
        )
        output = self.o_proj(output.reshape(batch, query_length, -1))
        return output, (draft_key, draft_value)


class LEagleDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = LEagleAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_draft_kv=None,
        context_kv=None,
    ):
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            position_embeddings,
            attention_mask,
            past_draft_kv,
            context_kv,
        )
        hidden_states = hidden_states + attention
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present
