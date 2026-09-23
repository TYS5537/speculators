"""Typed training entrypoint; legacy scripts keep their original default policy."""

import argparse
import gc
import logging

import torch

from hs_connectors import HiddenStatesBackend
from speculators.config import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.models.eagle3.data import shift_batch
from speculators.models.eagle3.rotary_partial import install_partial_neox_rotary
from speculators.models.mtp.data import shift_batch_mtp
from speculators.models.utils import get_verifier_config
from speculators.train.config import TrainConfig
from speculators.train.dataloader import create_train_val_loaders
from speculators.train.distributed import (
    get_rank,
    is_distributed,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.draft_config import (
    DRAFT_ARCH_CONFIGS,
    MROPE_INVERSE_TOLERANCE,
    _maybe_apply_mrope_full_head_hack,
    create_transformer_layer_config,
    load_draft_transformer_layer_config,
)
from speculators.train.legacy_cli import (
    DSPARK_PAPER_BLOCK_SIZE,
    DSPARK_PAPER_EPOCHS,
    DSPARK_PAPER_LOSS_FN,
    DSPARK_PAPER_NUM_LAYERS,
    _apply_training_defaults,
    _checkpoint_freq,
    _validate_training_args,
    build_train_parser,
    finalize_train_args,
    parse_train_args,
)
from speculators.train.logger import (
    log_run_config,
    setup_metric_logger,
    setup_root_logger,
)
from speculators.train.model_config import validate_draft_init_args
from speculators.train.model_init import (
    build_draft_model,
)
from speculators.train.model_init import (
    build_from_config_only as _build_from_config_only,
)
from speculators.train.trainer import Trainer, TrainerConfig
from speculators.train.vocab_setup import parse_vocab_mappings

__all__ = [
    "DRAFT_ARCH_CONFIGS",
    "DSPARK_PAPER_BLOCK_SIZE",
    "DSPARK_PAPER_EPOCHS",
    "DSPARK_PAPER_LOSS_FN",
    "DSPARK_PAPER_NUM_LAYERS",
    "MROPE_INVERSE_TOLERANCE",
    "_apply_training_defaults",
    "_build_from_config_only",
    "_checkpoint_freq",
    "_maybe_apply_mrope_full_head_hack",
    "_validate_training_args",
    "build_draft_model",
    "build_train_parser",
    "create_transformer_layer_config",
    "finalize_train_args",
    "load_draft_transformer_layer_config",
    "main",
    "parse_train_args",
    "parse_vocab_mappings",
]

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    import random  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _resolve_checkpoint_recipe(cfg: TrainConfig) -> TrainConfig:
    """An implicitly selected recipe must not switch an MMuse checkpoint to upstream."""
    if (
        not cfg.draft.from_pretrained
        or not cfg.provenance
        or cfg.provenance.get("training_recipe", "default") != "default"
        or cfg.training_recipe == "legacy"
    ):
        return cfg
    saved = SpeculatorModelConfig.from_pretrained(cfg.draft.from_pretrained)
    if saved.speculators_model_type != "mmuse":
        return cfg
    return cfg.with_checkpoint_recipe("mmuse", "legacy")


def main(cfg: TrainConfig):  # noqa: C901
    # Phase-1 adapter: the model layer still consumes a flat vars(args)-shaped
    # dict via **kwargs, so flatten the typed config back into a namespace here.
    # New code should read cfg.<group>.<field> directly and must NOT add new
    # args.* accesses below this line.
    cfg = _resolve_checkpoint_recipe(cfg)
    args = argparse.Namespace(**cfg.flatten())
    provided = {key for key, source in cfg.provenance.items() if source != "default"}
    vars(args)["_provided_model_config_dests"] = provided
    validate_draft_init_args(argparse.ArgumentParser(), args, provided)

    # Set random seed for reproducibility
    set_seed(args.seed, args.deterministic_cuda)

    # Setup logging
    setup_root_logger()
    setup_metric_logger(
        loggers=args.logger, run_name=args.run_name, output_dir=args.log_dir
    )

    # Setup distributed training
    maybe_setup_distributed()

    # Publish train config to metric backends that support hyperparameter logging
    log_run_config(cfg)

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
    # Write the reproducibility artifacts (run.yaml + train_command.txt) next to
    # the checkpoints at rank 0 only, so every checkpoint carries the resolved
    # config that produced it.
    if get_rank() == 0:
        cfg.save(args.save_path)

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

    target_config = get_verifier_config(
        args.verifier_name_or_path, trust_remote_code=args.trust_remote_code
    )
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
    # migrated to MMuse. Use the resolved class for preprocessing/trainer policy.
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
    # from_train_args is the live runtime consumer of the backend's (mirrored)
    # train-args, read off the flattened namespace. hs_connectors stays
    # argparse-based and standalone so vLLM can use it without speculators; that
    # is why the backend's train-args are mirrored into the pydantic schema rather
    # than the plugin depending on pydantic. test_backend_reconciliation.py keeps
    # the mirror complete so nothing read here was dropped during resolution.
    transfer = backend_cls.from_train_args(args, args.data_path)

    train_loader, val_loader = create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=hidden_states_dtype,
        noise_std=args.noise_std,
        legacy_data=args.legacy_data,
        pretokenized_text_only=args.dsv4_external_arrow,
        transfer=transfer,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        verifier_name_or_path=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        generation_validation_retries=(
            args.generation_validation_retries
            if args.training_recipe == "upstream"
            else None
        ),
        max_consecutive_generation_failures=args.max_consecutive_generation_failures,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=preprocess,
        train_data_ratio=args.train_data_ratio,
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
        gradient_checkpointing=args.gradient_checkpointing,
        activation_checkpointing=args.activation_checkpointing,
        training_recipe=args.training_recipe,
        max_steps=args.max_steps,
    )
    trainer = Trainer(draft_model, trainer_config, train_loader, val_loader)

    # Run training
    trainer.run_training()

    # Cleanup
    del trainer, draft_model
    gc.collect()
    acc = torch.accelerator.current_accelerator(check_available=True)
    if acc is not None:
        torch.get_device_module(acc.type).empty_cache()
    maybe_destroy_distributed()
