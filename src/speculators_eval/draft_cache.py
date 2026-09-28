"""Per-sample draft context reuse; no model/checkpoint or training changes.

The current DFlash/MMuse fusion is token-local and each decoder projects the
same immutable target context into K/V. Only confirmed context may survive a
proposal: neither the masked anchor placeholder nor the synthetic query block
is reusable, even when all draft tokens are accepted. Their replacements must
come from the verifier's real hidden states on the next round.
"""

import torch
from transformers import DynamicCache


def validate_draft_cache(draft) -> None:
    """Fail closed for position-dependent RoPE or an unknown attention layout."""
    from speculators.models.dflash.model_definitions import (  # noqa: PLC0415
        Qwen3DFlashAttention,
    )

    if draft.training:
        raise ValueError("Draft KV reuse requires an eval-mode model")
    rope_type = getattr(draft.rotary_emb, "rope_type", None)
    # Dynamic/longrope can change the rotation of ALREADY cached keys. Do not
    # silently reuse those keys or assume that staying under the limit is safe.
    if rope_type not in {"default", "linear", "yarn", "llama3"}:
        raise ValueError(f"Draft KV reuse does not support RoPE type {rope_type!r}")
    for index, layer in enumerate(draft.layers):
        attention = layer.self_attn
        if type(attention) is not Qwen3DFlashAttention or attention.layer_idx != index:
            raise ValueError("Draft KV reuse requires native DFlash attention layers")


class DraftContextCache:
    """Cache fused confirmed context and its native per-layer, rotated K/V.

    This object belongs to one generate_one call, never a model or a worker.
    Use an unconfigured DynamicCache so sliding-attention layers retain the
    exact same full key layout and mask as the reference evaluator.
    """

    def __init__(self):
        self.kv = DynamicCache()
        self.fused = None
        self.length = 0

    def prepare(self, draft, hidden_states, start):
        """Return full conditioning context and just the uncached K/V input."""
        if torch.is_grad_enabled() or draft.training:
            raise ValueError("Draft KV reuse is inference-only")
        if start < self.length or hidden_states.shape[1] != start + 1:
            raise ValueError("Draft KV context must be an append-only confirmed prefix")
        if self.kv.get_seq_length() != self.length:
            raise ValueError("Draft KV cache contains uncommitted proposal states")
        # hidden_states includes the original zero placeholder at `start`.
        # Keep that row and its position in this round's attention unchanged.
        suffix = draft._fuse_target_hidden(  # noqa: SLF001
            hidden_states[:, self.length :, :]
        )
        fused = suffix if self.fused is None else torch.cat([self.fused, suffix], dim=1)
        return fused, suffix

    def commit(self, fused, start):
        """Discard the placeholder and ALL synthetic block K/V, not just rejects."""
        self.kv.crop(start)
        self.fused = fused[:, :start, :]
        self.length = start
