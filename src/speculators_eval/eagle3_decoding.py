"""Single-chain EAGLE3 proposals using the repository's trained decoder weights."""

import torch
from transformers import DynamicCache


@torch.inference_mode()
def propose_eagle3(draft, *, hidden_states, input_ids, position_ids, num_tokens):
    """Rebuild the verified prefix, then feed back only our own predicted tokens.

    Feature at position i is paired with token i+1, including the target's known
    anchor. This is the same shift and recurrent state used by EAGLE3 training.
    """
    if draft.training or draft.uses_sliding_window_attn:
        raise ValueError("EAGLE3 proposals require eval mode and full attention")
    length = hidden_states.shape[1]
    if input_ids.shape != (1, length + 1) or position_ids.shape != (1, length):
        raise ValueError("Expected verified prefix features and tokens plus one anchor")
    if num_tokens < 1:
        raise ValueError("num_tokens must be positive")
    state = hidden_states
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
    state = draft.fc(state)
    tokens = input_ids[:, 1:]
    positions = position_ids + 1
    cache = DynamicCache()
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
        tokens = logits.argmax(dim=-1)
        if draft.use_draft_vocab:
            tokens = tokens + draft.d2t[tokens]
        proposals.append(tokens)
        all_logits.append(logits)
        positions = positions[:, -1:] + 1
    return torch.cat(proposals, dim=1), torch.cat(all_logits, dim=1)
