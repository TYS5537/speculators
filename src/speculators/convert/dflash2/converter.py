"""Convert Z Lab DFlash2 checkpoints, including materialized selector codebooks.

The source contract is pinned in ``models/dflash2/model_definitions.py``. It
stores draft options either in ``dflash_config`` or at the top level, with the
nested values taking precedence. Both Parameter and Embedding codebook names
are accepted; the remaining trained tensors must match the local model exactly.
The conversion supports full-attention drafts. Source sliding-window masks use
per-position windows, unlike the local anchored-window training contract.
"""

import torch

from speculators.convert.dflash.converter import DFlashConverter
from speculators.models.dflash2 import DFlash2DraftModel, DFlash2SpeculatorConfig

_DFLASH2_FIELDS = (
    "conv_kernel_size",
    "conv_group_size",
    "selector_rank",
    "selector_top_k",
)


def is_dflash2_config(config: dict) -> bool:
    """Detect DFlash2 before the generic DFlash architecture prefix."""
    nested = config.get("dflash_config") or {}
    return any("DFlash2" in name for name in config.get("architectures") or []) or all(
        nested.get(field, config.get(field)) is not None for field in _DFLASH2_FIELDS
    )


class DFlash2Converter(DFlashConverter):
    """Reuse DFlash's verifier loading with the standalone DFlash2 architecture."""

    model_class = DFlash2DraftModel

    def _build_config(
        self,
        source_config: dict,
        base_model: str,
        aux_hidden_state_layer_ids: list[int] | None,
    ) -> DFlash2SpeculatorConfig:
        nested = source_config.get("dflash_config") or {}

        def value(name, default=None):
            return nested.get(name, source_config.get(name, default))

        dimensions = {name: value(name) for name in _DFLASH2_FIELDS}
        missing = [name for name, size in dimensions.items() if size is None]
        if missing:
            raise ValueError(
                f"DFlash2 checkpoint config is missing: {', '.join(missing)}"
            )
        # These source-side scaling variants are not implemented by the local
        # backbone. Reject them rather than silently changing model predictions.
        if (
            value("input_embedding_scale", 1.0) != 1.0
            or value("output_multiplier", 1.0) != 1.0
            or (value("final_logit_softcapping") or 0.0) > 0.0
        ):
            raise ValueError(
                "DFlash2 conversion does not support embedding/logit scaling"
            )
        if value("sample_from_anchor", False):
            raise ValueError(
                "External DFlash2 checkpoints require sample_from_anchor=False"
            )
        if source_config.get("is_causal"):
            raise ValueError("DFlash2 conversion requires non-causal full attention")

        target_layer_ids = value("target_layer_ids")
        if aux_hidden_state_layer_ids is None:
            if target_layer_ids is None:
                target_layers = source_config.get("num_target_layers")
                draft_layers = source_config.get("num_hidden_layers")
                if not target_layers or not draft_layers:
                    raise ValueError(
                        "DFlash2 conversion needs target_layer_ids or both "
                        "num_target_layers and num_hidden_layers"
                    )
                target_layer_ids = (
                    [target_layers // 2]
                    if draft_layers == 1
                    else [
                        round(1 + i * (target_layers - 4) / (draft_layers - 1))
                        for i in range(draft_layers)
                    ]
                )
            # Preserve every trained FC input, including the last target layer.
            # DFlash's legacy converter drops that layer; DFlash2 must not.
            aux_hidden_state_layer_ids = [index + 1 for index in target_layer_ids]

        source = {k: v for k, v in source_config.items() if k not in _DFLASH2_FIELDS}
        source["dflash_config"] = {
            **nested,
            "mask_token_id": value("mask_token_id"),
            "target_layer_ids": value("target_layer_ids"),
        }
        source["block_size"] = value("block_size", 16)
        base = super()._build_config(source, base_model, aux_hidden_state_layer_ids)
        if "sliding_attention" in base.transformer_layer_config.layer_types:
            raise ValueError(
                "External DFlash2 sliding-window masks are not supported by this "
                "converter; use a full-attention checkpoint"
            )
        values = base.to_dict()
        values.update(
            speculators_model_type="dflash2",
            architectures=["DFlash2DraftModel"],
            sliding_window_non_causal=True,
            sample_from_anchor=False,
            **dimensions,
        )
        values["speculators_config"]["algorithm"] = "dflash2"
        return DFlash2SpeculatorConfig(**values)

    def _remap_weights(self, weights, config, model) -> dict[str, torch.Tensor]:
        remapped = dict(super()._remap_weights(weights, config, model))
        for name in ("predecessor_codebook", "successor_codebook"):
            target = f"candidate_selector.{name}"
            source = f"{target}.weight"
            if source in remapped:
                if target in remapped:
                    raise ValueError(
                        f"Duplicate selector codebook keys: {source}, {target}"
                    )
                remapped[target] = remapped.pop(source)
        return remapped
