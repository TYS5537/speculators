"""Request-local EAGLE prefix caches; synthetic draft states never survive a round."""

from functools import partial

import torch
from transformers import DynamicCache

from speculators.models.leagle import LEagleDraftModel
from speculators.models.leagle.cache import ReusableDraftKV, StaticKVBuffer
from speculators.models.utils import validate_kv_reuse_rope
from speculators_eval.eagle3_decoding import propose_eagle3


def _validate_rope(draft):
    validate_kv_reuse_rope(
        draft.rotary_emb, hint="use --no-draft-kv-reuse for this checkpoint"
    )


class LEagleCachedProposer:
    """Reuse real target context projections, independently of per-loop draft KV."""

    def __init__(self, draft, *, capacity=None):
        _validate_rope(draft)
        self.draft = draft
        self.length = 0
        self.context_kv = []
        self.buffers = (
            [StaticKVBuffer(capacity) for _ in draft.layers]
            if draft.static_kv and capacity is not None
            else None
        )
        if draft.static_kv and self.buffers is None:
            raise ValueError(
                "Static verified-prefix KV requires an explicit request capacity"
            )
        self.rollout_buffers = (
            [ReusableDraftKV(capacity) for _ in range(draft.config.effective_depth)]
            if draft.reuse_rollout_buffers
            else None
        )

    @torch.inference_mode()
    def __call__(
        self,
        draft,
        *,
        hidden_states,
        input_ids,
        position_ids,
        num_tokens,
        temperature=0.0,
        return_logits=True,
    ):
        length = hidden_states.shape[1]
        if draft is not self.draft or draft.training:
            raise ValueError("LEAGLE cache belongs to one eval-mode draft and request")
        if length <= self.length:
            raise ValueError(
                "LEAGLE cached proposals require a growing verified prefix"
            )
        if input_ids.shape != (1, length + 1) or position_ids.shape != (1, length):
            raise ValueError(
                "Expected verified prefix features and tokens plus one anchor"
            )
        # Only newly VERIFIED target features enter this cache. The preceding
        # round's predicted states are never valid substitutes, even on acceptance.
        suffix = draft.hidden_norm(draft.fc(hidden_states[:, self.length :]))
        rope = draft.rotary_emb(suffix, position_ids[:, self.length :])
        updated = []
        for index, layer in enumerate(draft.layers):
            key, value = layer.self_attn.project_context(suffix, rope)
            if self.buffers is not None:
                key, value = self.buffers[index].append(key, value)
            elif self.context_kv:
                old_key, old_value = self.context_kv[index]
                key = torch.cat([old_key, key], dim=-2)
                value = torch.cat([old_value, value], dim=-2)
            updated.append((key, value))
        self.context_kv = updated
        self.length = length
        rollout_kwargs = {}
        if self.rollout_buffers is not None:
            for index, buffer in enumerate(self.rollout_buffers):
                buffer.begin_round(*self.context_kv[index % len(draft.layers)])
            rollout_kwargs["rollout_buffers"] = self.rollout_buffers
        return draft.propose(
            hidden_states,
            input_ids[:, -1:],
            position_ids=position_ids,
            num_tokens=num_tokens,
            temperature=temperature,
            return_logits=return_logits,
            prepared_context=(suffix[:, -1:], self.context_kv),
            **rollout_kwargs,
        )


def cached_proposal(draft, propose, *, capacity=None):
    """Create one cache per generation; custom proposal callbacks are unchanged."""
    if propose is propose_eagle3:
        _validate_rope(draft)
        return partial(propose_eagle3, prefix_cache=DynamicCache())
    if propose is None and isinstance(draft, LEagleDraftModel):
        return LEagleCachedProposer(draft, capacity=capacity)
    return propose
