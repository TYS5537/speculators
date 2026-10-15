"""Train CLI declaration, algorithm defaults, and argument validation."""

import argparse

from hs_connectors import HiddenStatesBackend
from speculators.models.metrics import resolve_loss_config
from speculators.models.mmuse.config import (
    mmuse_option_defaults,
    validate_mmuse_options,
)
from speculators.train.config.cli_fields import field_argument_kwargs
from speculators.train.config.schema import _GROUPS, _ROOT_FIELDS, TrainConfig
from speculators.train.draft_config import DRAFT_ARCH_CONFIGS
from speculators.train.mmuse_args import (
    add_mmuse_backbone_args,
    add_mmuse_correction_args,
)
from speculators.train.model_config import (
    DECODER_SHAPING_FLAGS,
    MMUSE_MODEL_CONFIG_FIELDS,
    PRETRAINED_MODEL_CONFIG_FLAGS,
    validate_draft_init_args,
)
from speculators.utils.argparse_utils import explicitly_provided_dests

DSPARK_PAPER_LOSS_FN = '{"ce": 0.1, "tv": 0.9}'
DSPARK_PAPER_BLOCK_SIZE = 7
DSPARK_PAPER_NUM_LAYERS = 5
DSPARK_PAPER_EPOCHS = 10


def _checkpoint_freq(value: str) -> float:
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError("--checkpoint-freq must be > 0")
    if fvalue > 1 and not fvalue.is_integer():
        raise argparse.ArgumentTypeError(
            f"--checkpoint-freq={fvalue} is not an integer. Values > 1 are treated "
            "as epoch counts and must be whole numbers."
        )
    return fvalue


# Explicit allowlist preserves the legacy flag surface and registration order.
# New config-only fields must not silently add flags (or change abbreviations) here.
_LEGACY_FIELDS = """
    verifier_name_or_path trust_remote_code speculator_type from_pretrained
    draft_config dry_run data_path hidden_states_backend vllm_endpoint on_missing
    on_generate request_timeout max_retries legacy_data save_path epochs lr
    train_data_ratio no_resume_from_checkpoint logger total_seq_len log_freq
    profile_freq
    log_dir run_name num_layers draft_arch draft_hidden_act draft_mrope_full_head_hack
    target_layer_ids target_hidden_state_format dsv4_external_arrow token_freq_path
    draft_vocab_size d2t_path t2d_path mask_token_id ttt_steps num_speculative_steps
    ttt_step_loss_decay loss_fn step_weight_beta seed hidden_states_dtype
    deterministic_cuda norm_before_residual embed_requires_grad norm_before_fc
    fc_norm norm_output block_size sample_from_anchor max_anchors dflash_decay_gamma
    per_position_loss_weight dpace_alpha
    conv_kernel_size conv_group_size selector_rank selector_top_k selector_loss_alpha
    markov_rank markov_head_type
    enable_confidence_head confidence_head_with_markov confidence_detach_features
    confidence_head_alpha confidence_length_alpha confidence_loss_weighting
    first_error_focal_alpha adaptive_loss ssal_curriculum ssal_curriculum_start
    ssal_curriculum_end draft_attn_impl num_depths down_sample_ratio
    down_sample_ratio_min sliding_window full_attention_indices
    sliding_window_non_causal num_workers prefetch_factor noise_std checkpoint_freq
    save_best activation_checkpointing activation_checkpoint_layers
    fsdp_shard scheduler_type
    scheduler_warmup_steps scheduler_warmup_ratio scheduler_total_steps
    scheduler_num_cosine_cycles optimizer weight_decay muon_lr muon_momentum
    muon_weight_decay muon_ns_steps muon_adjust_lr_fn
""".split()  # noqa: SIM905 -- Compact, ordered compatibility allowlist.

# Only differences from the schema belong here. Algorithm-derived defaults and
# validation still run in finalize_train_args, never during parser construction.
_LEGACY_DEFAULTS = {
    "verifier_name_or_path": None,
    "lr": 1e-4,
    "num_layers": 1,
    "loss_fn": "kl_div",
    "block_size": 8,
    "max_anchors": 3072,
    "per_position_loss_weight": "fixed-exp-decay",
    "sliding_window_non_causal": False,
}
_LEGACY_HELP = {
    "speculator_type": "Type of speculator model to train "
    "(eagle3, dflash, dflash2, dspark, mmuse, peagle, mtp); "
    "muse is a legacy alias for mmuse",
    "epochs": "Training epochs (default: 20; DSpark paper default: 10).",
    "num_layers": "Draft decoder layers (default: 1; DSpark paper default: 5).",
    "block_size": "Draft block size (default: 8; DSpark paper default: 7).",
    "max_anchors": "Maximum anchor positions for DFlash, DSpark, "
    "and P-EAGLE training (default: 3072).",
    "loss_fn": "Loss function specification. Pass a name for a single loss "
    "(kl_div, rkl, jsd, ce, tv, nla, lk_hybrid) or a JSON dict for a weighted "
    'combination, e.g. \'{"ce": 0.1, "tv": 0.9}\'. The DSpark default '
    "is the paper's CE=0.1, TV=0.9 combination.",
    "muon_lr": "LR for the Muon (2D weights) group. Only used with --optimizer muon. "
    "Defaults to 10*lr (and --lr defaults to 1e-4)",
    "dflash_decay_gamma": "Decay gamma for DFlash/DSpark loss weighting "
    "(default: 4.0; DSpark paper default: its block size).",
    "per_position_loss_weight": "Per-position loss weight option for D-PACE support "
    "(default: fixed-exp-decay).",
    "sliding_window_non_causal": "Use non-causal (bidirectional) masking within "
    "draft blocks for sliding window attention layers. Full attention layers are "
    "always bidirectional. Enabled by default for DFlash2; disabled otherwise.",
    "target_layer_ids": "(Optional) A (space separated) list of integer layer ids. "
    "Defaults to [2, num_hidden_layers // 2, num_hidden_layers - 3, "
    "num_hidden_layers]. "
    "Note: must be set explicitly if custom values were used to launch vllm",
    "target_hidden_state_format": "Opt-in DSV4 auxiliary-mean / post-hc_head "
    "teacher HS contract.",
    "dsv4_external_arrow": "Accept external tokenized text Arrow without this repo's "
    "data manifest. You confirm its tokenizer/template/masks match the current "
    "DSV4 target. Checks all rows structurally and caps token/mask prefixes "
    "at --total-seq-len before HS requests without rewriting Arrow/order. "
    "Never bypasses an existing manifest or the target HS/checkpoint contract.",
    "legacy_data": "DEPRECATED. Use the old data format which stores hidden states "
    "alongside token_ids and assistant_masks, in data_i.pt files. "
    "This option will be removed soon.",
    "activation_checkpointing": "Recompute DFlash/DSpark dense decoder layers during "
    "backward to save activation memory. Leaves correction/other heads and loss "
    "unchanged. Training only; independent of --fsdp-shard. Disabled by default.",
}


def build_train_parser() -> argparse.ArgumentParser:
    """Declare legacy options from shared fields without resolving a TrainConfig."""
    parser = argparse.ArgumentParser()
    mmuse_defaults = mmuse_option_defaults()
    fields = {name: TrainConfig.model_fields[name] for name in _ROOT_FIELDS}
    fields.update(
        {
            name: field
            for group in _GROUPS.values()
            for name, field in group.model_fields.items()
        }
    )
    overrides = {
        "verifier_name_or_path": {"required": True},
        "draft_arch": {"choices": list(DRAFT_ARCH_CONFIGS)},
        "hidden_states_backend": {
            "type": None,
            "choices": list(HiddenStatesBackend.registry),
        },
        "on_missing": {"type": None},
        "on_generate": {"type": None},
        "target_hidden_state_format": {"type": None},
        "per_position_loss_weight": {"type": None},
        "sliding_window_non_causal": {"action": argparse.BooleanOptionalAction},
        "activation_checkpointing": {"action": "store_true"},
        "checkpoint_freq": {"type": _checkpoint_freq},
    }
    for name in _LEGACY_FIELDS:
        field = fields[name]
        kwargs = field_argument_kwargs(field)
        kwargs["default"] = (
            _LEGACY_DEFAULTS[name]
            if name in _LEGACY_DEFAULTS
            else field.get_default(call_default_factory=True)
        )
        kwargs.update(overrides.get(name, {}))
        if name in _LEGACY_HELP:
            kwargs["help"] = _LEGACY_HELP[name]
        parser.add_argument("--" + name.replace("_", "-"), dest=name, **kwargs)
        # Keep these registrars live and at their original positions. In particular,
        # do not use resolution's import-time backend snapshot or instantiate MMuse.
        if name == "hidden_states_backend":
            for backend_cls in HiddenStatesBackend.registry.values():
                backend_cls.add_train_args(parser)
        elif name == "dpace_alpha":
            add_mmuse_backbone_args(parser, mmuse_defaults)
        elif name == "markov_head_type":
            add_mmuse_correction_args(parser, mmuse_defaults)
    return parser


def _apply_dspark_paper_defaults(args: argparse.Namespace, provided: set[str]) -> None:
    """Fill the DSpark/MMuse recipe in order, retaining explicit CLI values.

    Resolve block size before deriving gamma; the shared defaults and validation
    still run afterwards. This helper neither normalizes aliases nor tracks flags.
    """
    if "block_size" not in provided:
        args.block_size = DSPARK_PAPER_BLOCK_SIZE
    if "dflash_decay_gamma" not in provided:
        # The paper uses the proposal length gamma in w_k=exp(-(k-1)/gamma).
        args.dflash_decay_gamma = float(args.block_size)
    if "epochs" not in provided:
        args.epochs = DSPARK_PAPER_EPOCHS
    if "loss_fn" not in provided:
        args.loss_fn = DSPARK_PAPER_LOSS_FN
    if "num_layers" not in provided:
        args.num_layers = DSPARK_PAPER_NUM_LAYERS


def _apply_training_defaults(args: argparse.Namespace, provided: set[str]) -> None:
    """Apply the selected algorithm defaults without replacing explicit options."""
    if args.speculator_type in ("dspark", "mmuse"):
        _apply_dspark_paper_defaults(args, provided)
    elif args.speculator_type == "dflash2":
        if "num_layers" not in provided:
            args.num_layers = 5
        if "sliding_window_non_causal" not in provided:
            args.sliding_window_non_causal = True
        if args.sample_from_anchor is None:
            args.sample_from_anchor = False

    is_eagle3 = args.speculator_type == "eagle3"
    if args.draft_arch is None:
        args.draft_arch = "llama" if is_eagle3 else "qwen3"
    if args.norm_before_fc is None:
        args.norm_before_fc = is_eagle3
    if args.norm_output is None:
        args.norm_output = is_eagle3
    if args.muon_lr is None:
        args.muon_lr = 10 * args.lr


def _validate_dflash2_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    for field in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be > 0")
    if args.selector_loss_alpha < 0:
        parser.error("--selector-loss-alpha must be >= 0")


def _validate_training_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, provided: set[str]
) -> None:
    """Validate initialized arguments in the established CLI error order."""
    validate_draft_init_args(parser, args, provided)
    resolve_loss_config(args.loss_fn)
    if args.speculator_type == "dflash2":
        _validate_dflash2_args(parser, args)

    try:
        if args.from_pretrained:
            # The remaining values will come from the checkpoint, not parser
            # defaults. Check only explicitly supplied scalars at this stage.
            validate_mmuse_options(
                {
                    field: getattr(args, field)
                    for field in args._provided_model_config_dests  # noqa: SLF001
                    & MMUSE_MODEL_CONFIG_FIELDS
                },
                partial=True,
            )
        elif args.speculator_type == "mmuse":
            validate_mmuse_options(vars(args))
    except ValueError as error:
        parser.error(str(error))
    if args.per_position_loss_weight == "dpace":
        if args.loss_fn != "ce":
            parser.error("--per-position-loss-weight=dpace requires --loss-fn=ce")
        if not 0.0 < args.dpace_alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {args.dpace_alpha}")


def finalize_train_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Track explicit options, apply defaults, and validate parsed arguments."""
    if args.speculator_type == "muse":
        args.speculator_type = "mmuse"
    if args.dsv4_external_arrow and (
        args.target_hidden_state_format != "deepseek_v4_mean_hc_head"
        or args.speculator_type not in ("dspark", "mmuse")
        or args.legacy_data
    ):
        parser.error(
            "--dsv4-external-arrow requires DSV4 DSpark/MMuse training with Arrow data"
        )
    # This CLI-owned namespace metadata is shared with checkpoint reconciliation.
    args._provided_model_config_dests = explicitly_provided_dests(  # noqa: SLF001
        parser, PRETRAINED_MODEL_CONFIG_FLAGS, argv=argv
    )
    if (
        not args.from_pretrained
        and args.speculator_type != "mmuse"
        and (
            args._provided_model_config_dests  # noqa: SLF001
            & MMUSE_MODEL_CONFIG_FIELDS
        )
    ):
        parser.error(
            "Correction, backbone enhancement and Selector options now belong to "
            "MMuse; use --speculator-type mmuse. DFlash and DSpark select the "
            "baseline architectures."
        )

    # Preserve the shared CLI defaults for every other algorithm while making a
    # bare DSpark/MMuse runs inherit the DSpark paper training recipe. MMuse's
    # optional architecture extensions remain explicit opt-ins.
    dspark_default_dests = {
        "block_size",
        "dflash_decay_gamma",
        "epochs",
        "loss_fn",
        "num_layers",
        "sliding_window_non_causal",
    }
    dspark_provided = explicitly_provided_dests(parser, dspark_default_dests, argv=argv)
    _apply_training_defaults(args, dspark_provided)

    provided = explicitly_provided_dests(parser, DECODER_SHAPING_FLAGS, argv=argv)
    _validate_training_args(parser, args, provided)
    return args


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse training arguments from an explicit list or the process command line."""
    parser = build_train_parser()
    args = parser.parse_args() if argv is None else parser.parse_args(argv)
    return finalize_train_args(parser, args, argv=argv)
