"""Explicit checkpoint-to-candidate second-stage initialization."""

import logging
import time
from pathlib import Path

from speculators.model import SpeculatorModel
from speculators.models.leagle.candidate_training import prepare_candidate_training
from speculators.train.model_config import reconcile_pretrained_config_args
from speculators.utils.loading import is_config_only_dir

logger = logging.getLogger(__name__)


def validate_candidate_destination(source, destination):
    """Keep second-stage artifacts out of the source checkpoint tree."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.is_dir() and (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("Candidate-only training requires a separate save directory")


def build_candidate_training_model(args, config, t2d, d2t):
    """Load the original draft strictly, then explicitly add/freeze its head."""
    if config.speculators_model_type != "leagle":
        raise ValueError("Candidate-only training requires a LEAGLE checkpoint")
    if is_config_only_dir(args.from_pretrained):
        raise ValueError("Candidate-only training requires trained checkpoint weights")
    if not args.candidate_rerank or (
        config.candidate_head and not config.candidate_rerank
    ):
        raise ValueError("Candidate-only training requires candidate reranking")
    provided = set(getattr(args, "_provided_model_config_dests", set()))
    if "candidate_head" in provided and not args.candidate_head:
        raise ValueError("Candidate-only training cannot disable the candidate head")
    if not config.speculators_config.verifier.name_or_path:
        config.speculators_config.verifier.name_or_path = args.verifier_name_or_path
    new_head = not config.candidate_head
    # Validate the future architecture before loading large weights. Ordinary
    # --from-pretrained continues to reject adding/removing candidate matrices.
    planned = type(config).from_dict(config.to_dict())
    if new_head:
        updates = {
            name: getattr(args, name)
            for name in (
                "candidate_head_rank",
                "candidate_head_top_k",
                "candidate_head_loss_weight",
                "candidate_head_chunk_size",
            )
        }
        planned = type(config).from_dict(
            {
                **planned.to_dict(),
                **updates,
                "candidate_head": True,
                "candidate_rerank": True,
            }
        )
    reconcile_pretrained_config_args(args, planned, logger=logger)
    for current in (config, planned):
        current.transformer_layer_config._attn_implementation = args.draft_attn_impl  # noqa: SLF001
    model, loading_info = SpeculatorModel.from_pretrained(
        args.from_pretrained,
        config=config,
        t2d=t2d,
        d2t=d2t,
        verifier=args.verifier_name_or_path,
        output_loading_info=True,
    )
    if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
        raise ValueError("Source draft checkpoint has missing or incompatible weights")
    model.config = planned
    started = time.perf_counter()
    prepare_candidate_training(model, initialization=args.candidate_head_init)
    logger.info(
        "LEAGLE candidate-only training: source=%s, initialization=%s, rank=%d, "
        "top_k=%d, setup=%.3fs; backbone frozen, candidate KL only",
        args.from_pretrained,
        args.candidate_head_init if new_head else "checkpoint",
        planned.candidate_head_rank,
        planned.candidate_head_top_k,
        time.perf_counter() - started,
    )
    return model
