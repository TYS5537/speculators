"""Sequential correction and confidence heads for the DSpark draft model."""

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention, silu

__all__ = [
    "CausalCorrectionHead",
    "ConfidenceHead",
    "MarkovHead",
]


CorrectionCache = list[tuple[torch.Tensor, torch.Tensor]]


class _TinyCausalAttention(nn.Module):
    """Small causal self-attention with an inference-friendly K/V cache."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = value.shape
        return value.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        query = self._split_heads(self.q_proj(hidden_states))
        key = self._split_heads(self.k_proj(hidden_states))
        value = self._split_heads(self.v_proj(hidden_states))

        past_len = 0
        if cache is not None:
            past_key, past_value = cache
            past_len = past_key.shape[-2]
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)

        query_len = query.shape[-2]
        if past_len == 0:
            attn_mask = None
            is_causal = query_len > 1
        else:
            # Query i is the absolute position past_len + i.
            query_pos = past_len + torch.arange(
                query_len, device=query.device
            ).unsqueeze(-1)
            key_pos = torch.arange(key.shape[-2], device=query.device).unsqueeze(0)
            attn_mask = key_pos <= query_pos
            is_causal = False

        output = scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).contiguous().view(
            hidden_states.shape[0], query_len, -1
        )
        next_cache = (key, value) if use_cache else None
        return self.o_proj(output), next_cache


class _TinyCausalLayer(nn.Module):
    """Pre-norm attention/MLP block used by :class:`CausalCorrectionHead`."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(hidden_size)
        self.attention = _TinyCausalAttention(hidden_size, num_heads)
        self.post_attention_norm = nn.RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.down_proj = nn.Linear(4 * hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        attention_output, next_cache = self.attention(
            self.input_norm(hidden_states), cache, use_cache=use_cache
        )
        hidden_states = hidden_states + attention_output
        normed = self.post_attention_norm(hidden_states)
        mlp_output = self.down_proj(
            silu(self.gate_proj(normed)) * self.up_proj(normed)
        )
        return hidden_states + mlp_output, next_cache


class CausalCorrectionHead(nn.Module):
    """Causal head that predicts low-rank residual draft logits.

    The DFlash hidden state remains aligned with the current draft position while
    ``previous_token_embeddings`` is shifted right by one position. The output
    projection is zero-initialized so enabling the head starts as an exact no-op.
    """

    def __init__(
        self,
        *,
        input_hidden_size: int,
        token_embedding_size: int,
        correction_hidden_size: int,
        correction_rank: int,
        draft_vocab_size: int,
        num_layers: int = 1,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if correction_hidden_size <= 0:
            raise ValueError("correction_hidden_size must be positive")
        if correction_rank <= 0:
            raise ValueError("correction_rank must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.hidden_proj = nn.Linear(
            input_hidden_size, correction_hidden_size, bias=False
        )
        self.token_proj = nn.Linear(
            token_embedding_size, correction_hidden_size, bias=False
        )
        self.layers = nn.ModuleList(
            [
                _TinyCausalLayer(correction_hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.RMSNorm(correction_hidden_size)
        self.correction_down = nn.Linear(
            correction_hidden_size, correction_rank, bias=False
        )
        self.correction_up = nn.Linear(
            correction_rank, draft_vocab_size, bias=False
        )
        nn.init.zeros_(self.correction_up.weight)

    def forward(
        self,
        previous_token_embeddings: torch.Tensor,
        dflash_hidden: torch.Tensor,
        cache: CorrectionCache | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, CorrectionCache | None]:
        """Return ``(correction_logits, causal_states, next_cache)``."""
        if previous_token_embeddings.shape[:-1] != dflash_hidden.shape[:-1]:
            raise ValueError(
                "previous-token embeddings and DFlash hidden states must align, got "
                f"{previous_token_embeddings.shape} and {dflash_hidden.shape}"
            )
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, got {len(cache)}"
            )

        hidden_states = self.token_proj(previous_token_embeddings) + self.hidden_proj(
            dflash_hidden
        )
        next_cache: CorrectionCache = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[layer_idx]
            hidden_states, next_layer_cache = layer(
                hidden_states, layer_cache, use_cache=use_cache
            )
            if use_cache:
                assert next_layer_cache is not None  # noqa: S101
                next_cache.append(next_layer_cache)

        causal_states = self.output_norm(hidden_states)
        correction_logits = self.correction_up(
            silu(self.correction_down(causal_states))
        )
        return correction_logits, causal_states, next_cache if use_cache else None


class MarkovHead(nn.Module):
    """Low-rank sequential logit bias ``B = W1 @ W2``.

    ``W1`` indexes the verifier vocabulary (the previous token id); ``W2`` projects
    to the draft vocabulary so the bias adds onto the DFlash logits.
    """

    def __init__(
        self,
        *,
        verifier_vocab_size: int,
        draft_vocab_size: int,
        markov_rank: int,
        hidden_size: int,
        head_type: str = "vanilla",
    ) -> None:
        super().__init__()
        if markov_rank <= 0:
            raise ValueError(f"markov_rank must be > 0, got {markov_rank}")
        if head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        self.head_type = head_type
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(verifier_vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + markov_rank, markov_rank)
        elif head_type == "rnn":
            # Joint [gate; candidate; output] projection over [state; prev_emb; hidden].
            self.joint_proj = nn.Linear(2 * markov_rank + hidden_size, 3 * markov_rank)

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up W1 embeddings for the given previous-token ids."""
        return self.markov_w1(token_ids.long())

    def block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,  # [N, block_size]
        hidden_states: torch.Tensor,  # [N, block_size, hidden]
        prev_emb: torch.Tensor | None = None,  # [N, block_size, r]
    ) -> torch.Tensor:
        """Return the per-position logit bias, shape [N, block_size, draft_vocab]."""
        if prev_emb is None:
            prev_emb = self.prev_embeddings(prev_token_ids)
        prev_emb = prev_emb.to(self.markov_w2.weight.dtype)

        if self.head_type == "vanilla":
            return self.markov_w2(prev_emb)

        if self.head_type == "gated":
            hidden_states = hidden_states.to(prev_emb.dtype)
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([hidden_states, prev_emb], dim=-1))
            )
            return self.markov_w2(gate * prev_emb)

        # rnn: maintain a recurrent state across block positions.
        hidden_states = hidden_states.to(prev_emb.dtype)
        num_blocks, block_size, _ = prev_emb.shape
        state = prev_emb.new_zeros(num_blocks, self.markov_rank)
        outputs = []
        for k in range(block_size):
            z = torch.cat([state, prev_emb[:, k], hidden_states[:, k]], dim=-1)
            gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
            outputs.append(self.markov_w2(torch.tanh(out_raw)))
        return torch.stack(outputs, dim=1)


class ConfidenceHead(nn.Module):
    """Per-position acceptance-probability predictor (linear -> scalar logit)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)
