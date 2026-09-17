"""Training-side validation, imported only by the explicit DSV4 HS path."""

import logging
from pathlib import Path

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import (
    DEFAULT_LAYERS,
    inspect_checkpoint,
    validate_layers,
)
from speculators_dsv4.external_data import validate_external_arrow
from speculators_dsv4.preprocessing import DATA_MANIFEST, validate_data_manifest
from speculators_dsv4.training_contract import (
    read_training_contract,
    validate_draft_contract,
)

logger = logging.getLogger(__name__)


def _validate_training_data(args, report, rank, world_size):
    external = getattr(args, "dsv4_external_arrow", False)
    manifest = Path(args.data_path) / DATA_MANIFEST
    # An explicit external-data opt-in can accept an absent native manifest, but
    # must never silence a malformed/mismatched manifest that is actually present.
    if not external or manifest.exists() or manifest.is_symlink():
        validate_data_manifest(args.data_path, report)
    if external:
        logger.info("Checking external DSV4 Arrow on rank %s/%s", rank, world_size)
        checked = validate_external_arrow(
            args.data_path,
            report["config"]["vocab_size"],
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            "External Arrow structure checked: %s rows [%s, %s) of %s",
            checked["data_path"],
            checked["row_start"],
            checked["row_stop"],
            checked["row_count"],
        )
        if rank == 0:
            logger.warning(
                "External DSV4 Arrow provenance is USER-ASSERTED, not "
                "tokenizer/template verified. Source Arrow and row order are "
                "preserved; token/mask prefixes are capped at --total-seq-len "
                "before HS requests. No native "
                "data manifest is created. Source=%s, target=%s, signature=%s. "
                "The opt-in and paths are recorded in train_command.txt. Keep a "
                "separate HS directory if this is a different dataset.",
                checked["data_path"],
                report["model_path"],
                report["checkpoint_signature"],
            )


def prepare_training(args, *, rank=0, world_size=1):
    if args.speculator_type != "dspark":
        raise ValueError("The DSV4 target adapter currently supports DSpark only.")
    if args.hidden_states_backend != "file" or args.legacy_data:
        raise ValueError("DSV4 currently requires the file HS backend and Arrow data.")
    report = inspect_checkpoint(args.verifier_name_or_path)
    _validate_training_data(args, report, rank, world_size)
    saved = None
    if args.from_pretrained:
        from transformers import PretrainedConfig  # noqa: PLC0415

        saved, _ = PretrainedConfig.get_config_dict(args.from_pretrained)
        if saved.get("target_hidden_state_format") != HS_FORMAT:
            raise ValueError("Checkpoint was not trained with the DSV4 HS contract.")
        layers = saved["aux_hidden_state_layer_ids"]
        if args.target_layer_ids is not None and args.target_layer_ids != layers:
            raise ValueError(
                "Cannot change target HS layers on a restored draft checkpoint."
            )
        args.target_layer_ids = layers
    elif not args.draft_config:
        raise ValueError(
            "Use --draft-config for DSV4; do not inherit its MLA head geometry."
        )
    elif args.target_layer_ids is None:
        args.target_layer_ids = list(DEFAULT_LAYERS)
    validate_layers(args.target_layer_ids)
    if args.mask_token_id is None:
        args.mask_token_id = report["config"].get("dspark_noise_token_id", 128799)
    directory = args.hidden_states_path or Path(args.data_path) / "hidden_states"
    contract = read_training_contract(directory, report, args.target_layer_ids)
    if saved is not None:
        validate_draft_contract(saved, contract, source=args.from_pretrained)
    args.target_training_contract = contract
    return report
