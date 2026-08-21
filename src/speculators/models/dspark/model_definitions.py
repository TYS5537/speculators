"""Sequential correction, Markov, and confidence heads for DSpark."""

from typing import Literal

import torch
from torch import nn
from torch.nn.functional import embedding, linear, scaled_dot_product_attention, silu

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
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(hidden_states.shape[0], query_len, -1)
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
        mlp_output = self.down_proj(silu(self.gate_proj(normed)) * self.up_proj(normed))
        return hidden_states + mlp_output, next_cache


class CausalCorrectionHead(nn.Module):
    """Predict a gated hidden or logit residual for the parallel DFlash output.

    Each block position combines the previous-token embedding, the current DFlash
    hidden state, and a learned block-position embedding.  Causal attention carries
    those features across the block.

    ``output_mode="hidden"`` preserves the baseline path: the returned residual is
    added to the DFlash hidden state before the model's full LM-head projection.
    No-grad rollout can equivalently cache ``LMHead @ correction_up`` and add a
    low-rank vocabulary residual to one parallel base projection.
    ``output_mode="logits"`` additionally encodes the previous position's
    target or rollout logits and returns a low-rank vocabulary bias that is added
    to ordinary DFlash base logits, analogous to the Markov head. Optional
    auxiliary hidden output and corrected-hidden feedback let logit mode retain
    and propagate a trainable pre-LM representation without changing its token
    output path. The caller may alternatively project that corrected hidden for
    the current token before adding the vocabulary residual.
    """

    def __init__(
        self,
        *,
        input_hidden_size: int,
        token_embedding_size: int,
        block_size: int,
        correction_hidden_size: int,
        correction_rank: int,
        num_layers: int = 1,
        num_heads: int = 8,
        gate_bias: float = 0.0,
        output_mode: Literal["hidden", "logits"] = "hidden",
        draft_vocab_size: int | None = None,
        enable_hidden_auxiliary: bool = False,
        enable_hidden_feedback: bool = False,
    ) -> None:
        super().__init__()
        if correction_hidden_size <= 0:
            raise ValueError("correction_hidden_size must be positive")
        if correction_rank <= 0:
            raise ValueError("correction_rank must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if output_mode not in ("hidden", "logits"):
            raise ValueError(f"Unsupported correction output mode: {output_mode!r}")
        if output_mode == "logits" and (
            draft_vocab_size is None or draft_vocab_size <= 0
        ):
            raise ValueError("draft_vocab_size must be positive for logits Correction")

        self.output_mode = output_mode
        self.draft_vocab_size = draft_vocab_size
        self.enable_hidden_feedback = enable_hidden_feedback
        self.hidden_proj = nn.Linear(
            input_hidden_size, correction_hidden_size, bias=False
        )
        self.token_proj = nn.Linear(
            token_embedding_size, correction_hidden_size, bias=False
        )
        self.position_embedding = nn.Embedding(block_size, correction_hidden_size)
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
            correction_rank,
            input_hidden_size if output_mode == "hidden" else int(draft_vocab_size),
            bias=False,
        )
        self.previous_logits_down: nn.Linear | None = None
        self.previous_logits_proj: nn.Linear | None = None
        if output_mode == "logits":
            # Markov-like W1/W2 factorization around the causal head:
            #   previous probs[V] -> rank -> correction state -> rank -> bias[V].
            # Keeping W1 separate from the zero-initialized output W2 makes the
            # previous-logit feature available from the first training step.
            self.previous_logits_down = nn.Linear(
                int(draft_vocab_size), correction_rank, bias=False
            )
            self.previous_logits_proj = nn.Linear(
                correction_rank, correction_hidden_size, bias=False
            )
        self.hidden_feedback_proj: nn.Linear | None = None
        if enable_hidden_feedback:
            self.hidden_feedback_proj = nn.Linear(
                input_hidden_size, correction_hidden_size, bias=False
            )
        self.auxiliary_hidden_up: nn.Linear | None = None
        self.auxiliary_hidden_gate: nn.Linear | None = None
        if output_mode == "logits" and (
            enable_hidden_auxiliary or enable_hidden_feedback
        ):
            self.auxiliary_hidden_up = nn.Linear(
                correction_rank, input_hidden_size, bias=False
            )
            self.auxiliary_hidden_gate = nn.Linear(correction_hidden_size, 1)
        self.residual_gate = nn.Linear(correction_hidden_size, 1)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.correction_up.weight)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.constant_(self.residual_gate.bias, gate_bias)
        if self.auxiliary_hidden_up is not None:
            nn.init.zeros_(self.auxiliary_hidden_up.weight)
        if self.auxiliary_hidden_gate is not None:
            nn.init.zeros_(self.auxiliary_hidden_gate.weight)
            nn.init.constant_(self.auxiliary_hidden_gate.bias, gate_bias)

        # Inference-only products of LM-head and Correction hidden up-projection
        # weights. They are deliberately ordinary attributes: checkpoints keep
        # the original factorized weights, while each inference worker lazily
        # materializes device/dtype-specific fused projections once. Hidden mode
        # uses ``correction_up``; logits mode uses ``auxiliary_hidden_up`` when
        # dual hidden/logit correction is enabled.
        self._lm_fusion_signature: tuple[object, ...] | None = None
        self._lm_fusion_weight: torch.Tensor | None = None

    def encode_previous_distribution(
        self,
        previous_logits_mask: torch.Tensor,
        *,
        previous_logits: torch.Tensor | None = None,
        candidate_ids: torch.Tensor | None = None,
        candidate_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compress a dense or Top-K previous distribution before Correction.

        The sparse rank path is exactly equivalent to multiplying a dense
        Top-K-supported probability vector by ``previous_logits_down`` while
        avoiding an ``[..., vocab]`` intermediate.
        """
        uses_dense = previous_logits is not None
        uses_sparse = candidate_ids is not None or candidate_logits is not None
        if uses_dense == uses_sparse:
            raise ValueError("Provide exactly one dense or sparse logit representation")
        if uses_sparse and (candidate_ids is None or candidate_logits is None):
            raise ValueError("Sparse logit features require candidate IDs and logits")

        if uses_dense:
            assert previous_logits is not None  # noqa: S101
            if previous_logits.shape[:-1] != previous_logits_mask.shape:
                raise ValueError("Dense previous logits and mask must align")
            if previous_logits.shape[-1] != int(self.draft_vocab_size):
                raise ValueError("Dense previous logits use the wrong vocabulary size")
            probabilities = torch.softmax(previous_logits.float(), dim=-1)
        else:
            assert candidate_ids is not None  # noqa: S101
            assert candidate_logits is not None  # noqa: S101
            if candidate_ids.shape != candidate_logits.shape:
                raise ValueError("Previous candidate IDs and logits must align")
            if candidate_ids.shape[:-1] != previous_logits_mask.shape:
                raise ValueError("Sparse previous logits and mask must align")
            probabilities = torch.softmax(candidate_logits.float(), dim=-1)

        if self.previous_logits_down is None:
            raise RuntimeError("Logit Correction is missing its rank projection")
        rank_dtype = self.previous_logits_down.weight.dtype
        if uses_dense:
            previous_rank = self.previous_logits_down(probabilities.to(rank_dtype))
        else:
            assert candidate_ids is not None  # noqa: S101
            rank_codebook = self.previous_logits_down.weight.transpose(0, 1)
            candidate_rank = embedding(candidate_ids.long(), rank_codebook)
            previous_rank = torch.einsum(
                "...k,...kr->...r",
                probabilities.to(candidate_rank.dtype),
                candidate_rank,
            )
        return previous_rank * previous_logits_mask.to(
            device=previous_rank.device,
            dtype=previous_rank.dtype,
        ).unsqueeze(-1)

    def _residual_from_causal_states(
        self,
        causal_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.correction_up(silu(self.correction_down(causal_states)))
        return residual * torch.sigmoid(self.residual_gate(causal_states))

    def clear_lm_head_fusion_cache(self) -> None:
        """Drop lazily materialized inference-only LM-head products."""
        self._lm_fusion_signature = None
        self._lm_fusion_weight = None

    def _lm_head_fusion_weights(
        self,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return cached ``LMHead @ hidden_up`` projection weights."""
        if torch.is_grad_enabled():
            raise RuntimeError("LM-head fusion is inference-only")

        if self.output_mode == "hidden":
            up_weight = self.correction_up.weight
        else:
            if self.auxiliary_hidden_up is None:
                raise RuntimeError(
                    "Logit Correction LM-head fusion requires its auxiliary "
                    "hidden residual"
                )
            up_weight = self.auxiliary_hidden_up.weight
        signature_weights = [lm_head_weight, up_weight]
        signature_items: list[object] = []
        for weight in signature_weights:
            try:
                version = weight._version  # noqa: SLF001
            except RuntimeError:
                # Parameters created inside inference_mode do not expose a
                # version counter; inference weights are immutable in practice.
                version = None
            signature_items.append(
                (
                    weight.data_ptr(),
                    version,
                    weight.device,
                    weight.dtype,
                    tuple(weight.shape),
                )
            )
        signature = tuple(signature_items)
        if (
            signature != self._lm_fusion_signature
            or self._lm_fusion_weight is None
        ):
            lm_for_up = lm_head_weight.detach().to(
                device=up_weight.device,
                dtype=up_weight.dtype,
            )
            self._lm_fusion_signature = signature
            self._lm_fusion_weight = torch.matmul(
                lm_for_up, up_weight.detach()
            )

        assert self._lm_fusion_weight is not None  # noqa: S101
        return self._lm_fusion_weight

    @torch.compiler.disable
    def fused_lm_head_residual(
        self,
        causal_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Project a hidden residual to logits through cached fused weights.

        This is algebraically equivalent to applying the LM-head weight to
        :meth:`auxiliary_hidden_residual`, but associates the two linear maps as
        ``(LMHead @ up) @ low_rank``. It therefore avoids materializing a full
        hidden-to-vocabulary projection at every sequential rollout position.
        """
        fused_weight = self._lm_head_fusion_weights(lm_head_weight)
        low_rank = silu(self.correction_down(causal_states))
        residual_logits = linear(
            low_rank.to(fused_weight.dtype),
            fused_weight,
        )
        gate_layer = (
            self.residual_gate
            if self.output_mode == "hidden"
            else self.auxiliary_hidden_gate
        )
        if gate_layer is None:
            raise RuntimeError(
                "Logit Correction LM-head fusion requires its auxiliary hidden gate"
            )
        gate = torch.sigmoid(gate_layer(causal_states))
        return residual_logits * gate.to(residual_logits.dtype)

    def auxiliary_hidden_residual(
        self,
        causal_states: torch.Tensor,
    ) -> torch.Tensor:
        """Project Correction states to an auxiliary DFlash-hidden residual."""
        if self.output_mode == "hidden":
            return self._residual_from_causal_states(causal_states)
        if self.auxiliary_hidden_up is None or self.auxiliary_hidden_gate is None:
            raise RuntimeError(
                "Auxiliary hidden residual was not enabled for logit Correction"
            )
        common_states = silu(self.correction_down(causal_states))
        delta_hidden = self.auxiliary_hidden_up(common_states)
        return delta_hidden * torch.sigmoid(self.auxiliary_hidden_gate(causal_states))

    def forward(
        self,
        previous_token_embeddings: torch.Tensor,
        dflash_hidden: torch.Tensor,
        block_positions: torch.Tensor,
        previous_logits: torch.Tensor | None = None,
        previous_logits_mask: torch.Tensor | None = None,
        previous_rank_features: torch.Tensor | None = None,
        previous_corrected_hidden: torch.Tensor | None = None,
        previous_corrected_hidden_mask: torch.Tensor | None = None,
        current_token_embeddings: torch.Tensor | None = None,
        cache: CorrectionCache | None = None,
        *,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, CorrectionCache | None]:
        """Return ``(delta, causal_states, next_cache)``.

        ``delta`` is hidden-sized in hidden mode and draft-vocabulary-sized in
        logits mode. The latter requires previous logits plus a validity mask so
        block position zero can use an explicit no-history feature.
        """
        prefix_shape = dflash_hidden.shape[:-1]
        if previous_token_embeddings.shape[:-1] != prefix_shape:
            raise ValueError("previous-token embeddings and DFlash hidden must align")
        if current_token_embeddings is not None and (
            current_token_embeddings.shape != dflash_hidden.shape
        ):
            raise ValueError(
                "current selector-token embeddings and DFlash hidden must align"
            )
        if block_positions.shape != prefix_shape:
            raise ValueError("block positions and DFlash hidden must align")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} cache entries, got {len(cache)}"
            )
        if self.hidden_feedback_proj is None:
            if (
                previous_corrected_hidden is not None
                or previous_corrected_hidden_mask is not None
            ):
                raise ValueError(
                    "previous corrected hidden is only valid when hidden feedback "
                    "is enabled"
                )
        else:
            if (
                previous_corrected_hidden is None
                or previous_corrected_hidden_mask is None
            ):
                raise ValueError(
                    "hidden-feedback Correction requires previous corrected hidden "
                    "and mask"
                )
            if previous_corrected_hidden.shape != dflash_hidden.shape:
                raise ValueError(
                    "previous corrected hidden and DFlash hidden must align"
                )
            if previous_corrected_hidden_mask.shape != prefix_shape:
                raise ValueError(
                    "previous corrected hidden mask and DFlash hidden must align"
                )
        needs_previous_logits = self.output_mode == "logits"
        has_dense_logits = previous_logits is not None
        has_compact_features = previous_rank_features is not None
        if not needs_previous_logits:
            if (
                has_dense_logits
                or has_compact_features
                or previous_logits_mask is not None
            ):
                raise ValueError(
                    "previous logits require logit-residual Correction mode"
                )
        else:
            if previous_logits_mask is None:
                raise ValueError(
                    "logit-aware Correction requires previous features and mask"
                )
            if has_dense_logits == has_compact_features:
                raise ValueError(
                    "Provide exactly one dense or compact previous-logit representation"
                )
            if previous_logits_mask.shape != prefix_shape:
                raise ValueError("previous logits mask and DFlash hidden must align")
            if has_dense_logits:
                assert previous_logits is not None  # noqa: S101
                expected_logits_shape = (*prefix_shape, int(self.draft_vocab_size))
                if previous_logits.shape != expected_logits_shape:
                    raise ValueError(
                        "Expected previous logits shape "
                        f"{expected_logits_shape}, got {tuple(previous_logits.shape)}"
                    )
                previous_rank_features = self.encode_previous_distribution(
                    previous_logits_mask,
                    previous_logits=previous_logits,
                )
            if self.output_mode == "logits":
                if self.previous_logits_down is None:
                    raise RuntimeError("Logit Correction rank projection is missing")
                expected_rank_shape = (
                    *prefix_shape,
                    self.previous_logits_down.out_features,
                )
                if (
                    previous_rank_features is None
                    or previous_rank_features.shape != expected_rank_shape
                ):
                    raise ValueError(
                        "Compact previous-rank features must have shape "
                        f"{expected_rank_shape}"
                    )

        dtype = self.hidden_proj.weight.dtype
        dflash_features = dflash_hidden.to(dtype)
        if current_token_embeddings is not None:
            # Fuse the Selector's current-token hypothesis before the existing
            # projection. Linearity makes this equivalent to a second projected
            # residual without another matrix multiplication or checkpoint weight.
            dflash_features = dflash_features + current_token_embeddings.to(dtype)
        hidden_states = (
            self.hidden_proj(dflash_features)
            + self.token_proj(previous_token_embeddings.to(dtype))
            + self.position_embedding(
                block_positions.to(device=dflash_hidden.device, dtype=torch.long)
            ).to(dtype)
        )
        if self.output_mode == "logits":
            assert self.previous_logits_proj is not None  # noqa: S101
            assert previous_rank_features is not None  # noqa: S101
            hidden_states = hidden_states + self.previous_logits_proj(
                previous_rank_features.to(dtype)
            )
        if self.hidden_feedback_proj is not None:
            assert previous_corrected_hidden is not None  # noqa: S101
            assert previous_corrected_hidden_mask is not None  # noqa: S101
            feedback_hidden = previous_corrected_hidden.to(dtype)
            feedback_hidden = feedback_hidden * previous_corrected_hidden_mask.to(
                device=dflash_hidden.device, dtype=dtype
            ).unsqueeze(-1)
            hidden_states = hidden_states + self.hidden_feedback_proj(feedback_hidden)

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
        delta = self._residual_from_causal_states(causal_states)
        return delta, causal_states, next_cache if use_cache else None


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
