"""Opt-in Qwen3 target storage and selected hidden-state capture for LEAGLE."""

from contextlib import ExitStack, contextmanager
from functools import partial

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

from speculators.models.leagle.cache import StaticKVBuffer


class VerifiedStaticLayer(DynamicLayer):
    """Preallocated storage with a dynamic *visible* prefix and exact rollback."""

    def __init__(self, capacity):
        super().__init__()
        self.buffer = StaticKVBuffer(capacity)

    def update(self, key_states, value_states, *_args, **_kwargs):
        self.keys, self.values = self.buffer.append(key_states, value_states)
        self.is_initialized = True
        self.dtype, self.device = key_states.dtype, key_states.device
        return self.keys, self.values

    def get_seq_length(self):
        return self.buffer.length

    def crop(self, max_length):
        length = self.buffer.length
        if not length:
            return
        if max_length < 0:
            max_length = length + max_length
        if max_length < 0:
            raise ValueError("Target cache crop cannot remove more than its prefix")
        if max_length >= length:
            return
        self.buffer.length = max_length
        self.keys = self.buffer.key[..., :max_length, :]
        self.values = self.buffer.value[..., :max_length, :]

    def reset(self):
        self.crop(0)


class VerifiedStaticCache(DynamicCache):
    def __init__(self, config, capacity):
        if config.model_type != "qwen3" or any(
            kind != "full_attention" for kind in config.layer_types
        ):
            raise ValueError("Static target KV requires full-attention Qwen3")
        super().__init__(config=config)
        self.layers = [VerifiedStaticLayer(capacity) for _ in self.layers]


class _SelectedHiddenTarget:
    def __init__(self, target, layer_ids):
        self.target = target
        self.layer_ids = tuple(dict.fromkeys(layer_ids))
        backbone = getattr(target, "model", None)
        self.layers = getattr(backbone, "layers", ())
        self.norm = getattr(backbone, "norm", None)
        if (
            target.config.model_type != "qwen3"
            or not self.layers
            or self.norm is None
            or any(index < 0 or index > len(self.layers) for index in self.layer_ids)
        ):
            raise ValueError(
                "Selected target states require a Qwen3 backbone and valid layer IDs"
            )
        self.states = {}

    def __getattr__(self, name):
        return getattr(self.target, name)

    def capture_input(self, index, _module, args, kwargs):
        self.states[index] = kwargs.get("hidden_states", args[0] if args else None)

    def capture_output(self, index, _module, _args, output):
        self.states[index] = output

    def __call__(self, *args, **kwargs):
        self.states.clear()
        output = self.target(*args, **{**kwargs, "output_hidden_states": False})
        if any(
            not isinstance(self.states.get(index), torch.Tensor)
            for index in self.layer_ids
        ):
            raise RuntimeError(
                "Target did not expose the requested Qwen3 hidden states"
            )
        # Preserve HF's numbering: slot 0 is the embedding stream, slot N is
        # the final normalized state. Only these selected tensors are retained.
        output.hidden_states = dict(self.states)
        self.states.clear()
        return output


@contextmanager
def selected_hidden_target(target, layer_ids, *, enabled):
    if not enabled:
        yield target
        return
    selected = _SelectedHiddenTarget(target, layer_ids)
    with ExitStack() as stack:
        for index in selected.layer_ids:
            if index == len(selected.layers):
                handle = selected.norm.register_forward_hook(
                    partial(selected.capture_output, index)
                )
            else:
                handle = selected.layers[index].register_forward_pre_hook(
                    partial(selected.capture_input, index), with_kwargs=True
                )
            stack.callback(handle.remove)
        try:
            yield selected
        finally:
            selected.states.clear()
