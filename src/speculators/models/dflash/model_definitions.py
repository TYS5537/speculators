from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.functional import embedding
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


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    *,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Apply a causal grouped convolution without crossing draft blocks."""
    if hidden_states.shape[-1] != num_groups * group_size:
        raise ValueError("Grouped-conv hidden size does not match its groups")
    if hidden_states.shape[-2] % block_size != 0:
        raise ValueError("Grouped-conv sequence length must be divisible by block_size")
    expected_delta_shape = (*hidden_states.shape[:-1], taps, num_groups)
    if delta.shape != expected_delta_shape:
        raise ValueError(
            f"Expected dynamic coefficients {expected_delta_shape}, "
            f"got {tuple(delta.shape)}"
        )

    blocks = hidden_states.reshape(-1, block_size, num_groups, group_size)
    dynamic = delta.reshape(-1, block_size, taps, num_groups)
    coefficients = base.reshape(1, 1, taps, num_groups, group_size) + (
        dynamic.unsqueeze(-1)
    )
    output = coefficients[:, :, 0] * blocks
    for tap in range(1, taps):
        if tap >= block_size:
            continue
        shifted = torch.cat(
            [
                torch.zeros_like(blocks[:, :tap]),
                coefficients[:, tap:, tap] * blocks[:, :-tap],
            ],
            dim=1,
        )
        output = output + shifted
    return output.reshape_as(hidden_states)


class DFlash2GroupedConv(nn.Module):
    """DFlash2 content-conditioned causal convolution around one sublayer."""

    def __init__(
        self,
        hidden_size: int,
        *,
        taps: int,
        group_size: int,
        block_size: int,
    ) -> None:
        super().__init__()
        if taps <= 0:
            raise ValueError(f"conv_kernel_size must be > 0, got {taps}")
        if group_size <= 0 or hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, taps, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
        )

    def reset_identity(self) -> None:
        """Start as an exact identity while leaving both paths trainable."""
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        return grouped_dynamic_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            block_size=self.block_size,
            num_groups=self.num_groups,
            group_size=self.group_size,
            taps=self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.taps,
            self.num_groups,
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], 0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2CandidateSelector(nn.Module):
    """Score a Top-K token using its predecessor and DFlash hidden state."""

    def __init__(
        self,
        *,
        hidden_size: int,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        rank: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"selector_rank must be > 0, got {rank}")
        if top_k <= 0 or top_k > draft_vocab_size:
            raise ValueError(
                "selector_top_k must be in [1, draft_vocab_size], "
                f"got {top_k} for vocab {draft_vocab_size}"
            )
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(torch.empty(verifier_vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(draft_vocab_size, rank))
        self.hidden_projection = nn.Linear(hidden_size, rank, bias=False)

    def reset_unary(self, initializer_range: float) -> None:
        """Initialize the selector to reproduce its unary Top-K logits."""
        with torch.no_grad():
            nn.init.normal_(
                self.predecessor_codebook,
                mean=0.0,
                std=initializer_range,
            )
            nn.init.normal_(
                self.successor_codebook,
                mean=0.0,
                std=initializer_range,
            )
            self.hidden_projection.weight.zero_()

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_ids.shape != unary_logits.shape:
            raise ValueError("Candidate IDs and unary logits must align")
        if candidate_ids.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("Candidates and hidden states must align")
        if previous_token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError("Previous-token IDs and hidden states must align")
        if candidate_ids.shape[-1] != self.top_k:
            raise ValueError(
                f"Expected selector_top_k={self.top_k}, got {candidate_ids.shape[-1]}"
            )

        predecessor = embedding(previous_token_ids.long(), self.predecessor_codebook)
        successor = embedding(candidate_ids.long(), self.successor_codebook)
        hidden = self.hidden_projection(hidden_states)
        transition = torch.einsum(
            "...r,...kr->...k",
            predecessor.to(hidden.dtype) * hidden,
            successor.to(hidden.dtype),
        )
        return unary_logits + transition.to(unary_logits.dtype)

    def score_lattice(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        predecessor_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return DFlash2's ``K_previous x K_current`` edge lattice."""
        if candidate_ids.shape != unary_logits.shape:
            raise ValueError("Candidate IDs and unary logits must align")
        if candidate_ids.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("Candidates and hidden states must align")
        if predecessor_ids.shape != candidate_ids.shape:
            raise ValueError("Predecessor and current candidate lattices must align")
        if candidate_ids.shape[-1] != self.top_k:
            raise ValueError(
                f"Expected selector_top_k={self.top_k}, got {candidate_ids.shape[-1]}"
            )

        predecessor = embedding(predecessor_ids.long(), self.predecessor_codebook)
        successor = embedding(candidate_ids.long(), self.successor_codebook)
        hidden = self.hidden_projection(hidden_states)
        transitions = torch.einsum(
            "...pr,...r,...cr->...pc",
            predecessor.to(hidden.dtype),
            hidden,
            successor.to(hidden.dtype),
        )
        return unary_logits.unsqueeze(-2) + transitions.to(unary_logits.dtype)


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
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
    ):
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
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(
            config=config,
            layer_idx=layer_idx,
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # type: ignore[arg-type]
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,  # type: ignore[arg-type]
        )
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
