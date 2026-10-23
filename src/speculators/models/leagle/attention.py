"""Masks for parallel training of independent autoregressive trajectories."""

import torch


def trajectory_mask(document_ids, position_ids, step, sliding_window=None):
    """KV layout: target context, then one sequence-sized block per token step.

    Data has already passed through EAGLE's shift_batch: context[q] describes
    token position_ids[q]-1, while input_ids[q] is the next known token. Each
    query may read its verified target prefix and only its own draft trajectory.
    Later TTT steps never reveal more teacher context.
    """
    length = document_ids.numel()

    def mask_mod(_b, _h, q, kv):
        index = kv % length
        is_context = kv < length
        same_doc = (document_ids[q] == document_ids[index]) & (document_ids[q] >= 0)
        context = is_context & (index <= q) & same_doc
        own_draft = (~is_context) & (index == q)
        if sliding_window is not None:
            query_pos = position_ids[q] + step
            context = context & (position_ids[index] - 1 > query_pos - sliding_window)
            draft_step = kv // length - 1
            own_draft = own_draft & (draft_step > step - sliding_window)
        # Keep the current draft diagonal even for padding, avoiding all-masked
        # softmax rows. Padding is excluded from both targets and context.
        return context | own_draft

    return mask_mod


def rollout_mask(context_positions, query_position, step, sliding_window=None):
    """Mask for one real draft trajectory (one query, no packed documents)."""
    context_length = context_positions.numel()

    def mask_mod(_b, _h, _q, kv):
        context_index = torch.clamp(kv, max=context_length - 1)
        key_position = torch.where(
            kv < context_length,
            context_positions[context_index],
            query_position - step + kv - context_length,
        )
        visible = key_position <= query_position
        if sliding_window is not None:
            visible = visible & (key_position > query_position - sliding_window)
        return visible

    return mask_mod
