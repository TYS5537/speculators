import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.train import cli as _train_cli
from speculators.train import draft_config as _draft_config
from speculators.train import legacy_cli as _legacy_cli
from speculators.train import model_init as _model_init
from speculators.train import vocab_setup as _vocab_setup
from speculators.train.cli import (
    DSPARK_PAPER_BLOCK_SIZE,  # noqa: F401 -- Backward-compatible script export.
    DSPARK_PAPER_EPOCHS,  # noqa: F401 -- Backward-compatible script export.
    DSPARK_PAPER_LOSS_FN,  # noqa: F401 -- Backward-compatible script export.
    DSPARK_PAPER_NUM_LAYERS,  # noqa: F401 -- Backward-compatible script export.
    _checkpoint_freq,  # noqa: F401 -- Backward-compatible script export.
)
from speculators.train.draft_config import (
    DRAFT_ARCH_CONFIGS,  # noqa: F401 -- Backward-compatible script export.
    MROPE_INVERSE_TOLERANCE,  # noqa: F401 -- Backward-compatible script export.
)
from speculators.train.model_config import (
    DECODER_SHAPING_FLAGS,  # noqa: F401 -- Backward-compatible script export.
    MMUSE_MODEL_CONFIG_FIELDS,  # noqa: F401 -- Backward-compatible script export.
    PRETRAINED_MODEL_CONFIG_FLAGS,  # noqa: F401 -- Backward-compatible script export.
    PRETRAINED_RUNTIME_CONFIG_FIELDS,  # noqa: F401 -- Backward-compatible script export.
    reconcile_pretrained_config_args,
    validate_draft_init_args,  # noqa: F401 -- Backward-compatible script export.
)

# Keep the existing private helper import path available to script consumers.
from speculators.train.model_config import (
    plan_pretrained_config_overrides as _plan_pretrained_config_overrides,  # noqa: F401
)

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _maybe_apply_mrope_full_head_hack(
    rope_params: dict,
    resolved_head_dim: int,
    enabled: bool,
) -> None:
    """Optionally rescale partial MRoPE settings to full-head semantics."""
    _draft_config._maybe_apply_mrope_full_head_hack(
        rope_params, resolved_head_dim, enabled, logger=logger
    )


def create_transformer_layer_config(
    verifier_name_or_path: str,
    num_layers: int,
    draft_arch: str,
    hidden_act: str | None,
    sliding_window: int,
    full_attention_indices: list[int],
    mrope_full_head_hack: bool = True,
    *,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    """Build a draft decoder config using the training entrypoint logger."""
    return _draft_config.create_transformer_layer_config(
        verifier_name_or_path,
        num_layers,
        draft_arch,
        hidden_act,
        sliding_window,
        full_attention_indices,
        mrope_full_head_hack,
        **({"trust_remote_code": True} if trust_remote_code else {}),
        logger=logger,
    )


def load_draft_transformer_layer_config(
    draft_config: str,
    verifier_name_or_path: str,
    *,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    """Load and reconcile a draft decoder config using the entrypoint logger."""
    return _draft_config.load_draft_transformer_layer_config(
        draft_config,
        verifier_name_or_path,
        logger=logger,
        **({"trust_remote_code": True} if trust_remote_code else {}),
    )


def _load_mappings(d2t_path, t2d_path, expected_draft_vocab_size: int | None):
    """Load paired mappings using the training entrypoint logger."""
    return _vocab_setup._load_mappings(
        d2t_path, t2d_path, expected_draft_vocab_size, logger=logger
    )


def _save_vocab_mapping_atomically(path: Path, values: np.ndarray) -> None:
    """Publish a complete numpy file, never a visible partially written cache."""
    _vocab_setup._save_vocab_mapping_atomically(path, values)


def _parse_vocab_mappings_local(args: argparse.Namespace):
    """Resolve local mappings using the training entrypoint logger."""
    return _vocab_setup._parse_vocab_mappings_local(args, logger=logger)


def parse_vocab_mappings(args: argparse.Namespace):
    """Resolve and distribute mappings using the training entrypoint logger."""
    return _vocab_setup.parse_vocab_mappings(args, logger=logger)


def _build_from_config_only(
    model_class: type[SpeculatorModel],
    path: str,
    t2d: torch.Tensor | None,
    d2t: torch.Tensor | None,
    verifier_name_or_path: str | None = None,
    draft_attn_impl: str | None = None,
    training_args: argparse.Namespace | None = None,
) -> SpeculatorModel:
    """Initialize from a saved config using the training entrypoint logger."""
    return _model_init.build_from_config_only(
        model_class,
        path,
        t2d,
        d2t,
        verifier_name_or_path,
        draft_attn_impl,
        training_args,
        logger=logger,
    )


def build_draft_model(
    args: argparse.Namespace,
    model_class: type[SpeculatorModel],
    t2d: torch.Tensor | None,
    d2t: torch.Tensor | None,
    draft_vocab_size: int | None,
) -> SpeculatorModel:
    """Resolve and initialize the draft using the training entrypoint logger."""
    return _model_init.build_draft_model(
        args, model_class, t2d, d2t, draft_vocab_size, logger=logger
    )


def main(args: argparse.Namespace):
    """Run the shared trainer with the historical script defaults."""
    from speculators.train.config import TrainConfig  # noqa: PLC0415

    if isinstance(args, TrainConfig):
        return _train_cli.main(args)
    cfg = TrainConfig.from_flat({**vars(args), "training_recipe": "legacy"})
    from speculators.utils.argparse_utils import (  # noqa: PLC0415
        explicitly_provided_dests,
    )

    parser = _legacy_cli.build_train_parser()
    provided = explicitly_provided_dests(parser, set(vars(args)))
    provided = provided | {"training_recipe"}
    # Keep explicit checkpoint overrides distinct from parser-derived defaults.
    cfg._provenance = dict.fromkeys(provided, "flag")
    cfg._argv = list(sys.argv)
    return _train_cli.main(cfg)


def _reconcile_pretrained_config_args(
    args: argparse.Namespace,
    config: PretrainedConfig,
) -> None:
    """Validate the complete checkpoint/CLI merge, then commit it to both objects."""
    reconcile_pretrained_config_args(args, config, logger=logger)


def parse_args():
    """Parse training arguments from the process command line."""
    from speculators.train.config import TrainConfig  # noqa: PLC0415

    if any(
        arg.split("=", 1)[0] in {"--config", "--dump-config", "--training-recipe"}
        for arg in sys.argv[1:]
    ):
        return TrainConfig.resolve()
    return _legacy_cli.parse_train_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


# RUN WITH:
# torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py
# for multi-GPU training (DDP by default)
# OR
# torchrun --standalone --nproc_per_node=<num_gpus>  scripts/train.py --fsdp-shard
# for FSDP sharded training (when model doesn't fit in a single GPU)
# OR
# python scripts/train.py
# for single GPU training
