"""Draft-model initialization from checkpoints, decoder configs, or CLI settings."""

import argparse
import logging

import torch

from speculators.config import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.models.utils import get_verifier_config
from speculators.train.draft_config import (
    create_transformer_layer_config,
    load_draft_transformer_layer_config,
)
from speculators.train.model_config import reconcile_pretrained_config_args
from speculators.train.utils import resolve_mask_token_id
from speculators.utils.loading import is_config_only_dir

_LOGGER = logging.getLogger(__name__)


def build_from_config_only(
    model_class: type[SpeculatorModel],
    path: str,
    t2d: torch.Tensor | None,
    d2t: torch.Tensor | None,
    verifier_name_or_path: str | None = None,
    draft_attn_impl: str | None = None,
    training_args: argparse.Namespace | None = None,
    *,
    logger: logging.Logger | None = None,
) -> SpeculatorModel:
    """Initialize a fresh draft from a saved speculator *config* (no weights).

    Mirrors the tail of ``from_training_args``: build the model from the full
    speculator config, load vocab mappings, and pull verifier weights -- but with
    no trained draft weights to restore (decoder weights are randomly initialized).
    """
    logger = _LOGGER if logger is None else logger
    config = SpeculatorModelConfig.from_pretrained(path)
    model_class = SpeculatorModel.registered_model_class_from_config(config)
    if training_args is not None:
        reconcile_pretrained_config_args(training_args, config, logger=logger)
    if draft_attn_impl is not None:
        # HF exposes its runtime attention backend through this private field.
        config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
            draft_attn_impl
        )
    speculators_config = getattr(config, "speculators_config", None)
    # Fall back to the CLI --verifier-name-or-path only when the saved config has
    # no verifier path -- either null or blanked to "". A real path in the config
    # takes precedence and the CLI value is ignored.
    if (
        verifier_name_or_path
        and speculators_config is not None
        and not getattr(speculators_config.verifier, "name_or_path", None)
    ):
        speculators_config.verifier.name_or_path = verifier_name_or_path
    model = model_class(config=config)
    if hasattr(model, "load_vocab_mappings"):
        model.load_vocab_mappings(t2d, d2t)  # type: ignore[attr-defined, operator]
    if hasattr(model, "load_verifier_weights"):
        model.load_verifier_weights()  # type: ignore[attr-defined, operator]
    return model


def build_draft_model(
    args: argparse.Namespace,
    model_class: type[SpeculatorModel],
    t2d: torch.Tensor | None,
    d2t: torch.Tensor | None,
    draft_vocab_size: int | None,
    *,
    logger: logging.Logger | None = None,
) -> SpeculatorModel:
    """Resolve the draft model from one of these sources:

    * ``--from-pretrained``: finetune existing weights, or -- when the path is a
      config-only directory -- initialize fresh weights from a full saved
      speculator config.
    * ``--draft-config``: take the decoder ``transformer_layer_config`` from a
      config file; build the rest of the speculator from the other CLI args.
    * neither: synthesize the decoder from the verifier config + CLI flags.

    MTP is special-cased: when not loading ``--from-pretrained``, it reuses the
    verifier's own decoder config as the draft ``transformer_layer_config`` and
    extracts the native MTP head weights from the verifier, so the decoder-shaping
    flags and ``--draft-config`` do not apply.
    """
    logger = _LOGGER if logger is None else logger
    if args.from_pretrained:
        if is_config_only_dir(args.from_pretrained):
            logger.info(
                "--from-pretrained points to a config-only directory ('%s'); "
                "initializing fresh draft weights from the saved speculator config.",
                args.from_pretrained,
            )
            return build_from_config_only(
                model_class,
                args.from_pretrained,
                t2d=t2d,
                d2t=d2t,
                verifier_name_or_path=args.verifier_name_or_path,
                draft_attn_impl=(
                    args.draft_attn_impl if args.speculator_type != "mtp" else None
                ),
                training_args=args,
                logger=logger,
            )
        if args.speculator_type != "mtp":
            # _attn_implementation is never serialized by HF configs, so re-apply
            # the CLI selection before construction -- mirroring from_training_args.
            # MTP is skipped: its from_training_args never sets the field and its
            # __init__ resolves its own default ("eager") when it is absent.
            config = SpeculatorModelConfig.from_pretrained(args.from_pretrained)
            reconcile_pretrained_config_args(args, config, logger=logger)
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                args.draft_attn_impl
            )
            return SpeculatorModel.from_pretrained(
                args.from_pretrained,
                config=config,
                t2d=t2d,
                d2t=d2t,
                verifier=args.verifier_name_or_path,
            )
        # MTP keeps its own attention default, but it still needs the same saved
        # architecture/explicit-override checks as the other checkpoint paths.
        config = SpeculatorModelConfig.from_pretrained(args.from_pretrained)
        reconcile_pretrained_config_args(args, config, logger=logger)
        return model_class.from_pretrained(
            args.from_pretrained,
            config=config,
            t2d=t2d,
            d2t=d2t,
            verifier=args.verifier_name_or_path,
        )

    if args.speculator_type == "mtp":
        # MTP uses the verifier's own decoder config as the draft
        # transformer_layer_config and extracts the native MTP head weights from
        # the verifier; the decoder-shaping flags and --draft-config do not apply,
        # and there is no draft mask token to resolve.
        transformer_layer_config = get_verifier_config(
            args.verifier_name_or_path,
            **(
                {"trust_remote_code": True}
                if getattr(args, "trust_remote_code", False)
                else {}
            ),
        )
    else:
        if args.draft_config:
            transformer_layer_config = load_draft_transformer_layer_config(
                args.draft_config,
                args.verifier_name_or_path,
                logger=logger,
                **(
                    {"trust_remote_code": True}
                    if getattr(args, "trust_remote_code", False)
                    else {}
                ),
            )
        else:
            full_attention_indices = args.full_attention_indices
            if not full_attention_indices:
                logger.info(
                    "All %d draft layers using sliding window attention "
                    "(window=%d). To use full attention on specific layers, "
                    "pass '--full-attention-indices <layer_ids>'.",
                    args.num_layers,
                    args.sliding_window,
                )

            transformer_layer_config = create_transformer_layer_config(
                verifier_name_or_path=args.verifier_name_or_path,
                num_layers=args.num_layers,
                draft_arch=args.draft_arch,
                hidden_act=args.draft_hidden_act,
                sliding_window=args.sliding_window,
                full_attention_indices=full_attention_indices,
                mrope_full_head_hack=args.draft_mrope_full_head_hack,
                **(
                    {"trust_remote_code": True}
                    if getattr(args, "trust_remote_code", False)
                    else {}
                ),
                logger=logger,
            )

        args.mask_token_id = resolve_mask_token_id(
            args.verifier_name_or_path,
            transformer_layer_config.vocab_size,
            args.mask_token_id,
            **(
                {"trust_remote_code": True}
                if getattr(args, "trust_remote_code", False)
                else {}
            ),
        )

    args.draft_vocab_size = draft_vocab_size
    return model_class.from_training_args(
        verifier_config=transformer_layer_config,
        t2d=t2d,
        d2t=d2t,
        **vars(args),
    )
