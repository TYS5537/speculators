"""KV-conditioned decoder with explicit, immutable draft-cache inputs.

Only token steps append cache entries. A caller owns one cache per logical
depth (loop, physical layer), even though all loops share the layer weights.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
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
from speculators.models.leagle.cache import StaticDraftKV
from speculators.models.leagle.sparse_attention import (
    TrajectoryAttention,
    trajectory_attention,
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
        past_draft_kv: DraftKV | StaticDraftKV | None = None,
        context_kv: DraftKV | None = None,
        return_context_kv: bool = False,
    ):
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
            if (self.training or torch.is_grad_enabled()) and not return_context_kv:
                raise ValueError("Precomputed LEAGLE context KV is inference-only")
            context_key, context_value = context_kv
            draft_key = self.k_norm(
                self.k_proj(hidden_states).view(batch, query_length, -1, self.head_dim)
            ).transpose(1, 2)
            query, draft_key = apply_rotary_pos_emb(
                query,
                draft_key,
                *(value[:, -query_length:] for value in position_embeddings),
            )
        draft_value = (
            self.v_proj(hidden_states)
            .view(batch, query_length, -1, self.head_dim)
            .transpose(1, 2)
        )
        if past_draft_kv is not None and not isinstance(past_draft_kv, StaticDraftKV):
            draft_key = torch.cat([past_draft_kv[0], draft_key], dim=-2)
            draft_value = torch.cat([past_draft_kv[1], draft_value], dim=-2)
        output, present = self._attend(
            query,
            context_key,
            context_value,
            draft_key,
            draft_value,
            attention_mask,
            past_draft_kv,
        )
        output = self.o_proj(output.reshape(batch, query_length, -1))
        result = (output, present)
        return (*result, (context_key, context_value)) if return_context_kv else result

    def _attend(
        self, query, context_key, context_value, draft_key, draft_value, mask, cache
    ):
        if isinstance(mask, TrajectoryAttention):
            output = trajectory_attention(
                self, mask, query, context_key, context_value, draft_key, draft_value
            )
            return output, (draft_key, draft_value)
        if isinstance(cache, StaticDraftKV):
            key, value = cache.append_draft(
                context_key, context_value, draft_key, draft_value
            )
            present = cache
        else:
            # The immutable target prefix is never appended to the draft cache.
            key = torch.cat([context_key, draft_key], dim=-2)
            value = torch.cat([context_value, draft_value], dim=-2)
            present = (draft_key, draft_value)
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
            mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
        )
        return output, present


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
        return_context_kv=False,
        checkpoint_mlp=False,
    ):
        result = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            position_embeddings,
            attention_mask,
            past_draft_kv,
            context_kv,
            return_context_kv,
        )
        attention, present = result[:2]
        hidden_states = hidden_states + attention
        mlp_output = (
            checkpoint(
                self._mlp_forward,
                hidden_states,
                use_reentrant=False,
                preserve_rng_state=True,
            )
            if checkpoint_mlp and torch.is_grad_enabled()
            else self._mlp_forward(hidden_states)
        )
        hidden_states = hidden_states + mlp_output
        return hidden_states, present, *result[2:]

    def _mlp_forward(self, hidden_states):
        return self.mlp(self.post_attention_layernorm(hidden_states))
