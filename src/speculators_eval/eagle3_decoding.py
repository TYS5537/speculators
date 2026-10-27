"""Single-chain EAGLE3 proposals using the repository's trained decoder weights."""

import torch
from transformers import DynamicCache

from speculators.proposals.sampling import sample_from_logits, validate_temperature


def _fuse_target_hidden(draft, state):
    if draft.input_norm is not None:
        state = draft.input_norm(state)
    if draft.fc_norm is not None:
        state = torch.cat(
            [
                norm(chunk)
                for norm, chunk in zip(
                    draft.fc_norm, state.chunk(len(draft.fc_norm), dim=-1), strict=True
                )
            ],
            dim=-1,
        )
    return draft.fc(state)


@torch.inference_mode()
def propose_eagle3(
    draft,
    *,
    hidden_states,
    input_ids,
    position_ids,
    num_tokens,
    temperature=0.0,
    prefix_cache=None,
):
    """Update the verified prefix, then feed back only our own predicted tokens.

    Feature at position i is paired with token i+1, including the target's known
    anchor. This is the same shift and recurrent state used by EAGLE3 training.
    An optional request-local cache contains ONLY teacher-conditioned prefix KV.
    Synthetic rollout states must be discarded even when their tokens are accepted.
    """
    validate_temperature(temperature)
    if draft.training or draft.uses_sliding_window_attn:
        raise ValueError("EAGLE3 proposals require eval mode and full attention")
    length = hidden_states.shape[1]
    if input_ids.shape != (1, length + 1) or position_ids.shape != (1, length):
        raise ValueError("Expected verified prefix features and tokens plus one anchor")
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    cache = DynamicCache() if prefix_cache is None else prefix_cache
    cached_length = cache.get_seq_length()
    if cached_length >= length:
        raise ValueError("EAGLE3 cached proposals require a growing verified prefix")
    state = _fuse_target_hidden(draft, hidden_states[:, cached_length:])
    tokens = input_ids[:, cached_length + 1 :]
    positions = position_ids[:, cached_length:] + 1
    proposals, all_logits = [], []
    for _ in range(num_tokens):
        hidden = torch.cat([draft.embed_tokens(tokens), state], dim=-1)
        start = cache.get_seq_length()
        query_length = hidden.shape[1]
        cache_positions = torch.arange(
            start, start + query_length, device=hidden.device
        )
        allowed = (
            torch.arange(start + query_length, device=hidden.device)[None, :]
            <= cache_positions[:, None]
        )
        mask = allowed[None, None]
        if draft._attn_impl == "eager":  # noqa: SLF001
            mask = torch.zeros_like(mask, dtype=hidden.dtype).masked_fill(
                ~mask, -torch.inf
            )
        rope = draft.rotary_emb(hidden, positions)
        for layer in draft.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                cache_position=cache_positions,
                position_embeddings=rope,
                use_cache=True,
            )
        normalized = draft.norm(hidden[:, -1:])
        state = normalized if draft.config.norm_output else hidden[:, -1:]
        logits = draft.lm_head(normalized)
        tokens = sample_from_logits(logits, temperature)
        if draft.use_draft_vocab:
            tokens = tokens + draft.d2t[tokens]
        proposals.append(tokens)
        all_logits.append(logits)
        positions = positions[:, -1:] + 1
    # Keep only states computed from real target features. The next round fills
    # the newly verified suffix with teacher-conditioned states, not predictions.
    if prefix_cache is not None:
        cache.crop(length)
    return torch.cat(proposals, dim=1), torch.cat(all_logits, dim=1)
