"""Recompute decoder activations without mutating the caller's TTT cache."""

from transformers import DynamicCache


def checkpointed_decoder(
    layer, hidden, mask, positions, cache_positions, rope, previous, kwargs
):
    # Backward may execute this function again. A fresh cache on every call
    # prevents that recomputation from appending another token/TTT block.
    cache = DynamicCache()
    index = layer.self_attn.layer_idx
    if previous is not None:
        cache.update(*previous, index)
    output = layer(
        hidden,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        cache_position=cache_positions,
        position_embeddings=rope,
        **kwargs,
    )
    present = cache.layers[index]
    length = hidden.shape[1]
    return output, (present.keys[..., -length:, :], present.values[..., -length:, :])
