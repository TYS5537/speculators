from typing import TYPE_CHECKING, NamedTuple

import torch
from transformers.activations import ACT2FN
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    eager_attention_forward,
)
from typing_extensions import Unpack

if TYPE_CHECKING:
    from collections.abc import Callable


# Local copy of rotate_half to avoid dependency on internal transformers functions
def _rotate_half(x):
    """Rotates half the hidden dims of the input (local implementation)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,  # noqa: ARG001
    unsqueeze_dim=1,
):
    """Apply rotary position embeddings (local implementation)."""

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Implements the custom attention which injects the target models
    # hidden states into the kv cache.
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,  # type: ignore[operator]
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads  # type: ignore[operator]
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.k_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.v_proj = nn.Linear(
            config.hidden_size,  # type: ignore[arg-type]
            config.num_key_value_heads * self.head_dim,  # type: ignore[operator]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,  # type: ignore[operator]
            config.hidden_size,  # type: ignore[arg-type]
            bias=config.attention_bias,  # type: ignore[arg-type]
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.sliding_window = (
            config.sliding_window
            if hasattr(config, "layer_types")
            and config.layer_types is not None
            and config.layer_types[layer_idx] == "sliding_attention"  # type: ignore[index]
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Instead of computing the k and v matricies from the hidden states,
        # the target_hidden is injected into the kv cache, (shape is context
        # length + block size)
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        # This is the main difference from the usual attention mechanism.
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        # note the length becomes context length + block size
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,  # type: ignore[arg-type]
        )

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
        hidden_states = residual + hidden_states  # type: ignore[operator]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states  # type: ignore[operator,return-value]


class GlmDFlashRotaryEmbedding(nn.Module):
    """Partial rotary embedding used by GLM/DeepSeek-style MLA attention."""

    def __init__(self, config):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        rope_params = getattr(config, "rope_parameters", {}) or {}
        base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = self.inv_freq.to(device=x.device)
        freqs = torch.einsum("bl,d->bld", position_ids.float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class GlmDFlashMLP(nn.Module):
    def __init__(self, config, intermediate_size: int | None = None):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GlmDFlashMoE(nn.Module):
    """Training-friendly GLM MoE block with routed and shared experts."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = int(getattr(config, "n_routed_experts", 0))
        self.top_k = int(getattr(config, "num_experts_per_tok", 1))
        self.scoring_func = getattr(config, "scoring_func", "softmax")
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.routed_scaling_factor = float(
            getattr(config, "routed_scaling_factor", 1.0)
        )
        moe_intermediate_size = int(
            getattr(config, "moe_intermediate_size", config.intermediate_size)
        )
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                GlmDFlashMLP(config, moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                GlmDFlashMLP(config, moe_intermediate_size)
                for _ in range(int(getattr(config, "n_shared_experts", 0)))
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_experts <= 0:
            return hidden_states

        # TODO(glm): This is a correctness-first MoE implementation for initial
        # training smoke tests. Replace with a grouped/fused expert kernel before
        # using large GLM expert counts in serious training runs.
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, orig_shape[-1])
        scores = self.gate(flat)
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(scores)
        else:
            scores = torch.softmax(scores, dim=-1)

        top_k = min(self.top_k, self.num_experts)
        weights, selected = torch.topk(scores, top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights * self.routed_scaling_factor

        output = flat.new_zeros(flat.shape)
        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(selected == expert_idx)
            if token_idx.numel() == 0:
                continue
            expert_out = expert(flat[token_idx])
            output[token_idx] += expert_out * weights[token_idx, slot_idx].unsqueeze(
                -1
            )

        for shared_expert in self.shared_experts:
            output = output + shared_expert(flat)

        return output.reshape(orig_shape)


class GlmDFlashDSAIndexer(nn.Module):
    """DSA module scaffold.

    vLLM's sparse DSA indexer is tied to inference KV-cache backends. For training
    we keep the module and config surface so weights/configs have a landing spot,
    while the first implementation runs dense MLA attention.
    """

    def __init__(self, config):
        super().__init__()
        self.index_topk = getattr(config, "index_topk", None)
        self.index_topk_freq = getattr(config, "index_topk_freq", 1)
        self.index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        self.index_topk_pattern = getattr(config, "index_topk_pattern", None)
        self.indexer_rope_interleave = getattr(
            config, "indexer_rope_interleave", False
        )
        self.enabled = self.index_topk is not None
        self.last_topk_indices: torch.Tensor | None = None

    @torch.no_grad()
    def forward(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        **_kwargs,
    ) -> torch.Tensor | None:
        if not self.enabled:
            self.last_topk_indices = None
            return None
        topk = min(int(self.index_topk), key.shape[-2])
        # TODO(glm): Training scaffold only. We compute the DSA candidate set used
        # by sparse MLA backends, but do not yet feed it into an attention mask or
        # sparse kernel. Full parity should follow vLLM's GLM/DeepSeek indexer.
        scores = torch.matmul(query.float(), key.float().transpose(-1, -2)).mean(dim=1)
        self.last_topk_indices = torch.topk(scores, topk, dim=-1).indices.detach()
        return self.last_topk_indices


class GlmDFlashMLAAttention(nn.Module):
    """GLM/DeepSeek-style MLA attention with DFlash target-hidden KV injection."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.is_causal = False
        self.num_key_value_groups = 1
        bias = getattr(config, "attention_bias", False)

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.qk_head_dim, bias=bias
            )
        else:
            self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = Qwen3RMSNorm(
                self.q_lora_rank, eps=config.rms_norm_eps
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=bias
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=bias
        )
        self.kv_a_layernorm = Qwen3RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=bias
        )
        self.indexer = GlmDFlashDSAIndexer(config)
        self.sliding_window = (
            getattr(config, "sliding_window", None)
            if getattr(config, "layer_types", None) is not None
            and config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def _q_proj(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        return q.view(*hidden_states.shape[:-1], self.num_heads, self.qk_head_dim)

    def _kv_proj(self, hidden_states: torch.Tensor):
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_c, k_pe = torch.split(
            kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_b = self.kv_b_proj(self.kv_a_layernorm(kv_c))
        kv_b = kv_b.view(
            *hidden_states.shape[:-1],
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value = torch.split(
            kv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        return k_nope, k_pe, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,  # noqa: ARG002
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if past_key_values is not None:
            # TODO(glm): Training path only. Add KV-cache semantics when/if vLLM
            # serving support is implemented for GLM-style DFlash-family drafts.
            raise NotImplementedError(
                "GLM DFlash draft training does not use KV cache."
            )

        bsz, q_len = hidden_states.shape[:-1]
        all_hidden = torch.cat([target_hidden, hidden_states], dim=1)
        q = self._q_proj(hidden_states)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        k_nope, k_pe, value = self._kv_proj(all_hidden)
        k_pe = k_pe.unsqueeze(-2).expand(-1, -1, self.num_heads, -1)

        cos, sin = position_embeddings
        q_pe_t = q_pe.transpose(1, 2)
        k_pe_t = k_pe.transpose(1, 2)
        q_pe_t, k_pe_t = apply_rotary_pos_emb(q_pe_t, k_pe_t, cos, sin)
        q = torch.cat([q_nope.transpose(1, 2), q_pe_t], dim=-1).contiguous()
        key = torch.cat([k_nope.transpose(1, 2), k_pe_t], dim=-1).contiguous()
        value = value.transpose(1, 2).contiguous()

        self.indexer(query=q, key=key)

        attn_fn: Callable = eager_attention_forward
        if (
            self.config._attn_implementation is not None  # noqa: SLF001
            and self.config._attn_implementation != "eager"  # noqa: SLF001
        ):
            attn_fn = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation  # noqa: SLF001
            ]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), attn_weights


class GlmDFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GlmDFlashMLAAttention(config=config, layer_idx=layer_idx)
        if layer_idx < getattr(config, "first_k_dense_replace", 0):
            self.mlp = GlmDFlashMLP(config)
        else:
            self.mlp = GlmDFlashMoE(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,  # noqa: ARG002
        past_key_value: Cache | None = None,
        output_attentions: bool | None = False,  # noqa: ARG002
        use_cache: bool | None = False,  # noqa: ARG002
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        assert hidden_states is not None  # noqa: S101
        assert target_hidden is not None  # noqa: S101
        assert position_embeddings is not None  # noqa: S101
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            past_key_values=past_key_value,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class GlmDSparkDecoderLayer(GlmDFlashDecoderLayer):
    """DSpark GLM backbone layer.

    The MLA/MoE/DSA block is shared with DFlash. DSpark-specific behavior is
    applied after the backbone in ``DSparkDraftModel`` via Markov/confidence heads.
    This class gives DSpark checkpoints and module traversal an explicit GLM
    decoder type instead of only exposing the DFlash class name.
    """
    # TODO(glm): This remains a semantic alias until DSpark needs layer-local
    # behavior. The actual DSpark logic still lives after the backbone in
    # DSparkDraftModel.forward().


class DFlashModelComponents(NamedTuple):
    decoder_layer_class: type
    norm_class: type
    rotary_emb_class: type
    no_split_module: str


dflash_model_classes: dict[str, DFlashModelComponents] = {
    "qwen3": DFlashModelComponents(
        Qwen3DFlashDecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
        "Qwen3DFlashDecoderLayer",
    ),
    "glm_moe_dsa": DFlashModelComponents(
        GlmDFlashDecoderLayer,
        Qwen3RMSNorm,
        GlmDFlashRotaryEmbedding,
        "GlmDFlashDecoderLayer",
    ),
}

dspark_model_classes: dict[str, DFlashModelComponents] = {
    "glm_moe_dsa": DFlashModelComponents(
        GlmDSparkDecoderLayer,
        Qwen3RMSNorm,
        GlmDFlashRotaryEmbedding,
        "GlmDSparkDecoderLayer",
    ),
}


def resolve_dflash_model_components(
    model_type: str, algorithm: str | None = None
) -> DFlashModelComponents:
    """Resolve decoder components for DFlash-family algorithms."""
    if algorithm == "dspark" and model_type in dspark_model_classes:
        return dspark_model_classes[model_type]
    if model_type not in dflash_model_classes:
        raise ValueError(
            f"Unsupported DFlash-family draft model_type {model_type!r}. "
            f"Available: {sorted(dflash_model_classes)}"
        )
    return dflash_model_classes[model_type]
