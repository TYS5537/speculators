"""Create or load decoder configurations for draft-model initialization."""

import logging
import warnings
from copy import deepcopy

import transformers
from packaging import version
from transformers import LlamaConfig, PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators.models.utils import (
    get_verifier_config,
    resolve_draft_intermediate_size,
)

_LOGGER = logging.getLogger(__name__)

DRAFT_ARCH_CONFIGS: dict[str, type] = {
    "llama": LlamaConfig,
    "qwen3": Qwen3Config,
}
MROPE_INVERSE_TOLERANCE = 1e-6


def _maybe_apply_mrope_full_head_hack(
    rope_params: dict,
    resolved_head_dim: int,
    enabled: bool,
    *,
    logger: logging.Logger,
) -> None:
    """Optionally rescale partial MRoPE settings to full-head semantics."""
    if "mrope_section" not in rope_params:
        return

    inherited_partial = float(rope_params.get("partial_rotary_factor", 1.0))
    if enabled and inherited_partial < 1.0:
        old_section = list(rope_params["mrope_section"])
        inv = 1.0 / inherited_partial
        if abs(inv - round(inv)) > MROPE_INVERSE_TOLERANCE:
            raise ValueError(
                "mrope_full_head_hack cannot rescale mrope_section because "
                f"1/partial_rotary_factor={inv} is not an integer."
            )
        scale = int(round(inv))
        new_section = [int(x) * scale for x in old_section]
        if 2 * sum(new_section) != resolved_head_dim:
            raise ValueError(
                "mrope_full_head_hack rescaling produced inconsistent "
                f"mrope_section {new_section}: 2*sum={2 * sum(new_section)} "
                f"but head_dim={resolved_head_dim}."
            )
        rope_params["mrope_section"] = new_section
        rope_params["partial_rotary_factor"] = 1.0
        logger.warning(
            "MRoPE full-head hack applied: partial_rotary_factor "
            f"{inherited_partial} -> 1.0, mrope_section {old_section} -> "
            f"{new_section}."
        )
    elif not enabled and inherited_partial < 1.0:
        logger.warning(
            "mrope_full_head_hack=False with partial_rotary_factor="
            f"{inherited_partial} < 1.0 can cause HF trainer / vLLM "
            "partial-rotation mismatch."
        )


def create_transformer_layer_config(  # noqa: C901
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str,
    hidden_act: str | None,
    sliding_window: int,
    full_attention_indices: list[int],
    mrope_full_head_hack: bool = True,
    *,
    logger: logging.Logger | None = None,
) -> PretrainedConfig:
    logger = _LOGGER if logger is None else logger
    if draft_arch not in DRAFT_ARCH_CONFIGS:
        raise ValueError(
            f"Unknown draft architecture: {draft_arch}. "
            f"Available: {list(DRAFT_ARCH_CONFIGS.keys())}"
        )

    if draft_arch not in ("llama", "qwen3"):
        warnings.warn(
            f"Draft architecture '{draft_arch}' is not yet supported in vLLM. "
            "The trained model may not be usable for inference in vLLM. "
            "Consider using 'llama' or 'qwen3' for full vLLM compatibility.",
            stacklevel=2,
        )

    config_class = DRAFT_ARCH_CONFIGS[draft_arch]
    verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)

    # For multimodal models (Qwen3VL, etc.), extract text_config
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config

    if getattr(verifier_config, "model_type", None) == "deepseek_v4":
        raise ValueError(
            "DSV4 requires an explicit dense --draft-config, not target MLA geometry."
        )

    hidden_act = (
        hidden_act
        or getattr(verifier_config, "hidden_act", None)
        or getattr(verifier_config, "hidden_activation", None)
    )
    if hidden_act is None:
        raise AttributeError(
            f"{type(verifier_config).__name__} has neither 'hidden_act' "
            "nor 'hidden_activation'"
        )

    head_dim = getattr(verifier_config, "head_dim", None)
    num_attention_heads = verifier_config.num_attention_heads
    num_key_value_heads = verifier_config.num_key_value_heads

    if (
        head_dim
        and verifier_config.hidden_size % num_attention_heads != 0
        and verifier_config.hidden_size % head_dim == 0
    ):
        num_attention_heads = verifier_config.hidden_size // head_dim
        if num_attention_heads % num_key_value_heads != 0:
            num_key_value_heads = num_attention_heads
    resolved_head_dim = head_dim or verifier_config.hidden_size // num_attention_heads

    if full_attention_indices and (
        min(full_attention_indices) < 0 or max(full_attention_indices) >= num_layers
    ):
        raise ValueError(
            "Full attention indices must be valid draft layer ids "
            "in range [0, num_layers)."
        )
    layer_types = [
        "full_attention" if i in full_attention_indices else "sliding_attention"
        for i in range(num_layers)
    ]

    config = config_class(
        vocab_size=verifier_config.vocab_size,
        hidden_size=verifier_config.hidden_size,
        intermediate_size=resolve_draft_intermediate_size(verifier_config),
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_act=hidden_act,
        max_position_embeddings=verifier_config.max_position_embeddings,
        initializer_range=verifier_config.initializer_range,
        rms_norm_eps=verifier_config.rms_norm_eps,
        head_dim=head_dim,
        tie_word_embeddings=False,
        sliding_window=sliding_window,
        use_sliding_window="sliding_attention" in layer_types,
        layer_types=layer_types,
    )

    # New rope parameters definition introduced in transformers 5.0
    if version.parse(transformers.__version__) >= version.parse("5.0.0"):
        if hasattr(verifier_config, "rope_parameters"):
            rope_params = deepcopy(verifier_config.rope_parameters)
            # Some verifiers (e.g. Laguna) use the nested per-layer-type rope format
            # {"full_attention": {...}, "sliding_attention": {...}} with no top-level
            # "rope_theta". The llama-style draft uses a single rope, so collapse it to
            # a flat default rope (preferring the sliding-attention theta).
            if isinstance(rope_params, dict) and "rope_theta" not in rope_params:
                sub = (
                    rope_params.get("sliding_attention")
                    or rope_params.get("full_attention")
                    or {}
                )
                if isinstance(sub, dict):
                    rope_params = dict(sub)
                    rope_params.setdefault("rope_type", "default")
                    rope_params.setdefault("rope_theta", 10000.0)
                else:
                    rope_params = {"rope_type": "default", "rope_theta": 10000.0}

            if isinstance(rope_params, dict):
                _maybe_apply_mrope_full_head_hack(
                    rope_params, resolved_head_dim, mrope_full_head_hack, logger=logger
                )
                # ``type`` is a legacy alias (only "mrope" on VL models) that
                # transformers strips during validation and that breaks vLLM's
                # config checks; drop it while keeping the real MRoPE fields.
                rope_params.pop("type", None)
                rope_params.pop("mrope_interleaved", None)
                # The verifier (e.g. Mistral) may use partial rotary embeddings,
                # but the draft model doesn't support partial_rotary_factor.
                # Only keep it for MRoPE configs that need it.
                if "mrope_section" not in rope_params:
                    rope_params.pop("partial_rotary_factor", None)
            config.rope_parameters = rope_params
    else:
        if hasattr(verifier_config, "rope_scaling"):
            rope_scaling = deepcopy(verifier_config.rope_scaling)
            if isinstance(rope_scaling, dict):
                _maybe_apply_mrope_full_head_hack(
                    rope_scaling, resolved_head_dim, mrope_full_head_hack, logger=logger
                )
                # Strip legacy fields for consistency with rope_parameters path
                rope_scaling.pop("type", None)
                rope_scaling.pop("mrope_interleaved", None)
                # Same partial_rotary_factor guard as the rope_parameters path.
                if "mrope_section" not in rope_scaling:
                    rope_scaling.pop("partial_rotary_factor", None)
            config.rope_scaling = rope_scaling
        config.rope_theta = getattr(verifier_config, "rope_theta", 10000.0)

    return config


def load_draft_transformer_layer_config(
    draft_config: str,
    verifier_name_or_path: str,
    *,
    logger: logging.Logger | None = None,
) -> PretrainedConfig:
    """Load the draft decoder ``transformer_layer_config`` from a config source.

    ``draft_config`` may be a HF hub id, a local directory containing a
    ``config.json``, or a path to a config JSON file. It is expected to hold a
    plain decoder config (``LlamaConfig`` for eagle3/peagle, ``Qwen3Config`` for
    dflash). If a full speculator config is given instead, its nested
    ``transformer_layer_config`` is extracted as a convenience.

    The decoder is reconciled against the verifier: ``hidden_size`` must match
    (draft/verifier hidden-size mismatch is not yet supported) and ``vocab_size``
    is aligned to the verifier's target vocabulary. The pruned draft vocabulary
    is controlled separately via ``--draft-vocab-size``.
    """
    logger = _LOGGER if logger is None else logger
    config_dict, _ = PretrainedConfig.get_config_dict(draft_config)
    if "transformer_layer_config" in config_dict:
        # A full speculator config was passed; use only the decoder definition.
        config_dict = config_dict["transformer_layer_config"]

    model_type = config_dict.get("model_type")
    if not model_type:
        raise ValueError(
            "--draft-config must define a 'model_type' (e.g. 'llama' for "
            "eagle3/peagle, 'qwen3' for dflash); none was found in the config "
            f"loaded from '{draft_config}'."
        )
    config_class: type[PretrainedConfig] = type(AutoConfig.for_model(model_type))
    draft_config_obj = config_class.from_dict(config_dict)

    verifier_config = get_verifier_config(verifier_name_or_path)
    if draft_config_obj.hidden_size != verifier_config.hidden_size:
        raise ValueError(
            f"--draft-config hidden_size ({draft_config_obj.hidden_size}) must match "
            f"the verifier hidden_size ({verifier_config.hidden_size}). Draft/verifier "
            "hidden-size mismatch is not yet supported."
        )
    if draft_config_obj.vocab_size != verifier_config.vocab_size:
        logger.warning(
            "Overriding --draft-config vocab_size (%s) with the verifier vocab_size "
            "(%s). Use --draft-vocab-size to control the pruned draft vocabulary.",
            draft_config_obj.vocab_size,
            verifier_config.vocab_size,
        )
        draft_config_obj.vocab_size = verifier_config.vocab_size
    return draft_config_obj
