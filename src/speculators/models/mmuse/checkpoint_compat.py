"""Load pre-FSDP MMuse scalar gates without changing their learned values."""

from collections.abc import Mapping

import torch
from safetensors import safe_open

SCALAR_GATE_NAMES = (
    "layer_fusion_gate",
    "context_hidden_gate",
    "correction_markov_scale",
)
_ADAM_MOMENTS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")


def _checkpoint_gate_shapes(state_dict, checkpoint_files):
    """Inspect only gate metadata; leave checkpoint resolution/loading to HF."""
    if state_dict is not None:
        return {
            name: tuple(state_dict[name].shape)
            for name in SCALAR_GATE_NAMES
            if name in state_dict
        }
    shapes = {}
    for path in checkpoint_files or ():
        if str(path).endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as checkpoint:
                checkpoint_keys = set(checkpoint.keys())
                for name in SCALAR_GATE_NAMES:
                    if name in checkpoint_keys:
                        shapes[name] = tuple(checkpoint.get_slice(name).get_shape())
        else:
            # Older HF checkpoints may use a PyTorch archive. Reading on meta
            # avoids a second in-memory copy of all model weights.
            from transformers.modeling_utils import load_state_dict  # noqa: PLC0415

            tensors = load_state_dict(path, map_location="meta", weights_only=True)
            shapes.update(_checkpoint_gate_shapes(tensors, None))
    return shapes


class MMuseCheckpointMixin:
    """Migrate exactly the three formerly zero-dimensional MMuse parameters."""

    def _scalar_gate_parameters(self):
        return {
            name: parameter
            for name in SCALAR_GATE_NAMES
            if (parameter := getattr(self, name, None)) is not None
            and parameter.shape == (1,)
        }

    def _prepare_checkpoint_state_dict(self, state_dict, *, prefix=""):
        state_dict = state_dict.copy()
        for name in self._scalar_gate_parameters():
            key = prefix + name
            if key in state_dict and state_dict[key].ndim == 0:
                state_dict[key] = state_dict[key].reshape(1)
        return state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Cover direct torch loads, including an MMuse model nested in a wrapper.
        state_dict.update(
            self._prepare_checkpoint_state_dict(state_dict, prefix=prefix)
        )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def _load_pretrained_model(
        cls, model, state_dict, checkpoint_files, *args, **kwargs
    ):
        # HF assigns parameters directly and checks shapes before torch's load
        # hooks. Temporarily match legacy scalar metadata, then restore the new
        # layout before the model is returned or any optimizer/FSDP is created.
        shapes = _checkpoint_gate_shapes(state_dict, checkpoint_files)
        legacy_names = [
            name
            for name in cls._scalar_gate_parameters(model)
            if shapes.get(name) == ()
        ]
        for name in legacy_names:
            parameter = model.get_parameter(name)
            parameter.data = parameter.data.reshape(())
        try:
            return super()._load_pretrained_model(
                model, state_dict, checkpoint_files, *args, **kwargs
            )
        finally:
            for name in legacy_names:
                parameter = model.get_parameter(name)
                parameter.data = parameter.data.reshape(1)

    def _prepare_optimizer_state_dict(self, state_dict, *, optimizer=None):
        """Reshape old Adam moments, leaving scalar step counters untouched.

        Local optimizer archives use parameter IDs; distributed archives use
        fully qualified parameter names. The ID mapping follows the same group
        order as Optimizer.load_state_dict, without changing group membership.
        """
        gates = self._scalar_gate_parameters()
        keys = set(gates)
        if optimizer is not None:
            identities = {id(parameter) for parameter in gates.values()}
            keys = {
                saved_id
                for saved_group, group in zip(
                    state_dict["param_groups"], optimizer.param_groups, strict=False
                )
                for saved_id, parameter in zip(
                    saved_group["params"], group["params"], strict=False
                )
                if id(parameter) in identities
            }
        result = state_dict.copy()
        if isinstance(state_dict.get("state"), Mapping):
            result["state"] = state_dict["state"].copy()
            for key in keys & state_dict["state"].keys():
                values = state_dict["state"][key].copy()
                for name in _ADAM_MOMENTS:
                    value = values.get(name)
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        values[name] = value.reshape(1)
                result["state"][key] = values
        else:
            # Also support PyTorch's flattened distributed optimizer format.
            for key in keys:
                for name in _ADAM_MOMENTS:
                    flat_key = f"state.{key}.{name}"
                    value = result.get(flat_key)
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        result[flat_key] = value.reshape(1)
        return result
