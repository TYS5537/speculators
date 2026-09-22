import argparse
import gc
import logging
import random
from pathlib import Path

import numpy as np
import torch
from transformers import PretrainedConfig

from hs_connectors import HiddenStatesBackend
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.models.eagle3.rotary_partial import install_partial_neox_rotary
from speculators.models.mtp.data import shift_batch_mtp
from speculators.models.utils import get_verifier_config
from speculators.train import cli as _train_cli
from speculators.train import draft_config as _draft_config
from speculators.train import model_init as _model_init
from speculators.train import vocab_setup as _vocab_setup
from speculators.train.cli import (
    DSPARK_PAPER_BLOCK_SIZE,  # noqa: F401 -- Backward-compatible script export.
    DSPARK_PAPER_EPOCHS,  # noqa: F401 -- Backward-compatible script export.
    DSPARK_PAPER_LOSS_FN,  # noqa: F401 -- Backward-compatible script export.
    DSPARK_PAPER_NUM_LAYERS,  # noqa: F401 -- Backward-compatible script export.
    _checkpoint_freq,  # noqa: F401 -- Backward-compatible script export.
)
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.distributed import (
    get_rank,
    is_distributed,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.draft_config import (
    DRAFT_ARCH_CONFIGS,  # noqa: F401 -- Backward-compatible script export.
    MROPE_INVERSE_TOLERANCE,  # noqa: F401 -- Backward-compatible script export.
)
from speculators.train.logger import setup_metric_logger, setup_root_logger
from speculators.train.model_config import (
    DECODER_SHAPING_FLAGS,  # noqa: F401 -- Backward-compatible script export.
    MUSE_MODEL_CONFIG_FIELDS,  # noqa: F401 -- Backward-compatible script export.
    PRETRAINED_MODEL_CONFIG_FLAGS,  # noqa: F401 -- Backward-compatible script export.
    PRETRAINED_RUNTIME_CONFIG_FIELDS,  # noqa: F401 -- Backward-compatible script export.
    reconcile_pretrained_config_args,
    validate_draft_init_args,  # noqa: F401 -- Backward-compatible script export.
)

# Keep the existing private helper import path available to script consumers.
from speculators.train.model_config import (
    plan_pretrained_config_overrides as _plan_pretrained_config_overrides,  # noqa: F401
)
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.utils import save_train_command

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
        logger=logger,
    )


def load_draft_transformer_layer_config(
    draft_config: str,
    verifier_name_or_path: str,
) -> PretrainedConfig:
    """Load and reconcile a draft decoder config using the entrypoint logger."""
    return _draft_config.load_draft_transformer_layer_config(
        draft_config, verifier_name_or_path, logger=logger
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


def main(args: argparse.Namespace):  # noqa: C901
    # Set random seed for reproducibility
    set_seed(args.seed, args.deterministic_cuda)

    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    maybe_setup_distributed()

    if args.fsdp_shard and not is_distributed():
        raise ValueError(
            "--fsdp-shard requires launching with torchrun/distributed training; "
            "otherwise parameters are not sharded."
        )

    # Install partial-neox rotary patch if not using full-head hack
    if not args.draft_mrope_full_head_hack:
        install_partial_neox_rotary()
        logger.info(
            "Installed partial-neox rotary patch for HF/vLLM RoPE alignment "
            "(draft_mrope_full_head_hack=False)"
        )
    if get_rank() == 0:
        save_train_command(args.save_path)

    if not hasattr(torch, args.hidden_states_dtype):
        raise ValueError(
            "--hidden-states-dtype must be a dtype attribute of torch. e.g. `bfloat16`"
        )
    hidden_states_dtype = getattr(torch, args.hidden_states_dtype)

    if hidden_states_dtype == torch.float16:
        raise NotImplementedError(
            "--hidden-states-dtype=float16 is not supported. "
            "float16 with torch.autocast requires gradient scaling (GradScaler) to "
            "prevent gradient underflow, which is not implemented. "
            "Use bfloat16 instead, which provides the same memory savings with "
            "better numerical stability and no gradient scaling required."
        )

    if args.speculator_type == "mtp":
        if args.draft_attn_impl != "simple_flex_attention":
            raise ValueError(
                "--draft-attn-impl is not configurable for MTP. "
                "Must be left with the default value ('simple_flex_attention')."
            )
        # MTP reuses the verifier's own decoder as the draft and extracts the
        # native MTP head weights from the verifier, so there are no vocab
        # mappings or draft mask token to resolve from the CLI. This works both
        # with --from-pretrained (a previously converted checkpoint) and without
        # it (weights are extracted from the verifier on the fly). The decoder
        # transformer_layer_config is resolved later in build_draft_model.
        d2t, t2d, draft_vocab_size = None, None, None
        args.mask_token_id = None
    else:
        d2t, t2d, draft_vocab_size = parse_vocab_mappings(args)

        if args.full_attention_indices and args.speculator_type == "mtp":
            raise ValueError(
                "--full-attention-indices is not supported for mtp draft models."
            )

    target_config = get_verifier_config(args.verifier_name_or_path)
    target_is_dsv4 = getattr(target_config, "model_type", None) == "deepseek_v4"
    use_dsv4_format = args.target_hidden_state_format == "deepseek_v4_mean_hc_head"
    if getattr(args, "dsv4_external_arrow", False) and not use_dsv4_format:
        raise ValueError("--dsv4-external-arrow is only supported for DSV4 training")
    if target_is_dsv4 != use_dsv4_format:
        raise ValueError(
            "DSV4 targets require --target-hidden-state-format "
            "deepseek_v4_mean_hc_head; "
            "other targets must use standard."
        )
    if use_dsv4_format:
        from speculators_dsv4.training import prepare_training  # noqa: PLC0415
        from speculators_dsv4.training_contract import (  # noqa: PLC0415
            distributed_validation,
        )

        distributed_validation(
            lambda: prepare_training(
                args,
                rank=get_rank(),
                world_size=torch.distributed.get_world_size()
                if is_distributed()
                else 1,
            ),
            torch.distributed if is_distributed() else None,
        )

    registry = SpeculatorModel.registry
    if registry is None or args.speculator_type not in registry:
        available = list(registry.keys()) if registry else []
        raise ValueError(
            f"Unknown speculator type: {args.speculator_type}. Available: {available}"
        )

    model_class = registry[args.speculator_type]

    draft_model = build_draft_model(args, model_class, t2d, d2t, draft_vocab_size)
    # Saved configs are authoritative, including enhanced legacy DSpark configs
    # migrated to Muse. Use the resolved class for preprocessing/trainer policy.
    model_class = type(draft_model)
    args.speculator_type = draft_model.config.speculators_model_type

    if (
        use_dsv4_format
        and draft_model.config.target_hidden_state_format
        != args.target_hidden_state_format
    ):
        raise ValueError("Restored draft has an incompatible target HS format.")

    # Get target layer IDs from the model (resolved at model level)
    num_target_layers = len(draft_model.target_layer_ids)  # type: ignore[arg-type]

    if args.speculator_type == "mtp":
        args.num_speculative_steps = draft_model.config.num_speculative_steps

    # Dry-run: persist an initialized checkpoint and exit before training so the
    # config/weights can be validated (e.g. in vLLM). The saved checkpoint can be
    # fed straight back via --from-pretrained to start training.
    if args.dry_run:
        # Save in hidden_states_dtype (bf16) for compact checkpoints.
        draft_model.to(hidden_states_dtype)
        if get_rank() == 0:
            logger.info(
                "[dry-run] Saving initialized checkpoint (%s) to '%s'",
                hidden_states_dtype,
                args.save_path,
            )
            draft_model.save_pretrained(args.save_path)
            logger.info(
                "[dry-run] Done. Validate this checkpoint, then train with "
                "'--from-pretrained %s'.",
                args.save_path,
            )
        maybe_destroy_distributed()
        return

    hidden_size = draft_model.config.transformer_layer_config.hidden_size

    # Setup dataloaders
    preprocess_fns = {
        "eagle3": shift_batch,
        "peagle": shift_batch,
        "mtp": shift_batch_mtp,
    }
    preprocess = preprocess_fns.get(args.speculator_type)

    backend_registry = HiddenStatesBackend.registry
    backend_cls = backend_registry[args.hidden_states_backend]
    transfer = backend_cls.from_train_args(args, args.data_path)

    train_loader, val_loader = create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=hidden_states_dtype,
        noise_std=args.noise_std,
        legacy_data=args.legacy_data,
        transfer=transfer,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        verifier_name_or_path=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=preprocess,
        train_data_ratio=args.train_data_ratio,
        pretokenized_text_only=getattr(args, "dsv4_external_arrow", False),
    )

    # Get trainer kwargs from model class
    train_call_kwargs, val_call_kwargs = model_class.get_trainer_kwargs(**vars(args))

    trainer_config = TrainerConfig(
        num_epochs=args.epochs,
        save_path=args.save_path,
        lr=args.lr,
        resume_from_checkpoint=not args.no_resume_from_checkpoint,
        train_call_kwargs=train_call_kwargs,
        val_call_kwargs=val_call_kwargs,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        muon_lr=args.muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        scheduler_type=args.scheduler_type,
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_warmup_ratio=args.scheduler_warmup_ratio,
        scheduler_total_steps=args.scheduler_total_steps,
        scheduler_num_cosine_cycles=args.scheduler_num_cosine_cycles,
        checkpoint_freq=args.checkpoint_freq,
        save_best=args.save_best,
        hidden_states_dtype=hidden_states_dtype,
        log_freq=args.log_freq,
        fsdp_shard=args.fsdp_shard,
        activation_checkpointing=args.activation_checkpointing,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    del trainer, draft_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    maybe_destroy_distributed()


def _reconcile_pretrained_config_args(
    args: argparse.Namespace,
    config: PretrainedConfig,
) -> None:
    """Validate the complete checkpoint/CLI merge, then commit it to both objects."""
    reconcile_pretrained_config_args(args, config, logger=logger)


def parse_args():
    """Parse training arguments from the process command line."""
    return _train_cli.parse_train_args()


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
