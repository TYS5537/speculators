"""Train CLI declaration, algorithm defaults, and argument validation."""

import argparse

from hs_connectors import HiddenStatesBackend
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
)
from speculators.models.metrics import resolve_loss_config
from speculators.models.muse.config import muse_option_defaults, validate_muse_options
from speculators.train.draft_config import DRAFT_ARCH_CONFIGS
from speculators.train.model_config import (
    DECODER_SHAPING_FLAGS,
    MUSE_MODEL_CONFIG_FIELDS,
    PRETRAINED_MODEL_CONFIG_FLAGS,
    validate_draft_init_args,
)
from speculators.train.muse_args import add_muse_backbone_args, add_muse_correction_args
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


def build_train_parser() -> argparse.ArgumentParser:
    """Declare training options without parsing or applying algorithm defaults."""
    parser = argparse.ArgumentParser()
    muse_defaults = muse_option_defaults()
    parser.add_argument("--verifier-name-or-path", type=str, required=True)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow executing code from HF Hub when loading the verifier's tokenizer.",
    )
    parser.add_argument(
        "--speculator-type",
        type=str,
        default="eagle3",
        help="Type of speculator model to train "
        "(eagle3, dflash, dspark, muse, peagle, mtp)",
    )
    parser.add_argument(
        "--from-pretrained",
        type=str,
        default="",
        help="Path or HF id of a pretrained draft. May also point to a "
        "local directory containing only a config.json, in which case a "
        "fresh draft is initialized from that full speculator config. Takes precedence "
        "over and is mutually exclusive with --draft-config and the decoder-shaping "
        "flags (--num-layers, --draft-arch, --draft-hidden-act, --sliding-window, "
        "--full-attention-indices).",
    )
    parser.add_argument(
        "--draft-config",
        type=str,
        default="",
        help="HF id, directory, or JSON path of a decoder config (LlamaConfig for "
        "eagle3/peagle, Qwen3Config for dflash) to use as the draft "
        "transformer_layer_config; the rest of the speculator is built from the other "
        "CLI args. Mutually exclusive with --from-pretrained and with the "
        "decoder-shaping flags (--num-layers, --draft-arch, --draft-hidden-act, "
        "--sliding-window, --full-attention-indices).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Build the speculator, initialize weights, save a checkpoint to "
        "--save-path, then exit before training. Useful to validate the config and "
        "weights (e.g. in vLLM) before launching a full run. Can be combined with "
        "--draft-config or --from-pretrained.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./output",
        help=(
            "Root data directory containing the preprocessed dataset, "
            "vocab mappings (d2t.npy, t2d.npy), token frequencies "
            "(token_freq.pt), and hidden states (default: ./output)"
        ),
    )
    backend_registry = HiddenStatesBackend.registry
    parser.add_argument(
        "--hidden-states-backend",
        choices=list(backend_registry.keys()),
        default="file",
        help=(
            "Hidden states transfer backend. Each backend may add its own "
            "CLI arguments (see below). Default: 'file'."
        ),
    )
    for backend_cls in backend_registry.values():
        backend_cls.add_train_args(parser)

    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help=(
            "vLLM endpoint address to use if generating hidden states on-demand."
            " Only required if `--on-missing=generate` and samples are missing."
            " Note: the vLLM instance must be configured to cache hidden states"
            " to a location that is accessible from the training instance. i.e."
            " on the same node, or a shared network drive. (Default: 'http://localhost:8000/v1')"
        ),
    )
    parser.add_argument(
        "--on-missing",
        choices=["generate", "skip", "warn", "raise"],
        default="generate",
        help=(
            "Dataloader behaviour when there are no cached hidden states for a sample."
            "Default: 'generate', which attempts to generate the hidden states on-"
            "demand using the provided vLLM endpoint. The other options skip the sample"
            ", skip and warn, or raise an error respectively."
        ),
    )
    parser.add_argument(
        "--on-generate",
        choices=["cache", "delete"],
        default="delete",
        help=(
            "Dataloader behaviour when a new hidden state has been generated"
            " (only applies if args.on_missing=='generate'). Default: 'delete', "
            "deletes hidden states once they are loaded. 'cache' will instead store"
            "the hidden states in the args.hidden_states_path. This can be used to "
            "enable hybrid online/offline training, with hidden states generated on the"
            "first epoch, and reused on subsequent epochs."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help=(
            "Timeout in seconds for each individual vLLM request "
            f"(default: {DEFAULT_REQUEST_TIMEOUT}). "
            "Only applies if --on-missing=generate."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Maximum number of retry attempts per vLLM request on failure "
            f"(default: {DEFAULT_MAX_RETRIES}). "
            "Only applies if --on-missing=generate."
        ),
    )
    parser.add_argument(
        "--legacy-data",
        action="store_true",
        help=(
            "DEPRECATED. Use the old data format which stores hidden states alongside "
            "token_ids and assistant_masks, in data_i.pt files. This option will be "
            "removed soon."
        ),
    )
    parser.add_argument("--save-path", type=str, default="./output/checkpoints")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs (default: 20; DSpark paper default: 10).",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-data-ratio", type=float, default=0.9)
    parser.add_argument("--no-resume-from-checkpoint", action="store_true")
    parser.add_argument(
        "--logger",
        type=str,
        default="",
        help=(
            "One of 'trackio', 'wandb', 'tensorboard', 'mlflow' or "
            "comma separated list."
        ),
    )
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument(
        "--log-freq",
        type=int,
        default=1,
        help="Log training metrics every N steps (default: 1)",
    )
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Draft decoder layers (default: 1; DSpark paper default: 5).",
    )
    parser.add_argument(
        "--draft-arch",
        type=str,
        default=None,
        choices=list(DRAFT_ARCH_CONFIGS.keys()),
        help="Architecture for draft decoder layers "
        "(default: 'llama' for eagle3, 'qwen3' otherwise).",
    )
    parser.add_argument(
        "--draft-hidden-act",
        type=str,
        default="silu",
        help="Activation function for draft decoder layers. Defaults to 'silu' for "
        "sigmoid linear unit. Qwen3 layers of dflash expect 'silu' activation for "
        "vLLM deployment. If another function is desired, set as a string or leave "
        "as None to automatically fall back to the verifier's activation function.",
    )
    parser.add_argument(
        "--draft-mrope-full-head-hack",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For MRoPE configs with partial_rotary_factor < 1, rescale "
            "mrope_section and set partial_rotary_factor=1.0 so HF training "
            "and vLLM inference use equivalent full-head rotary semantics."
        ),
    )
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) A (space separated) list of integer layer ids. Defaults to"
            "[2, num_hidden_layers // 2, num_hidden_layers - 3, num_hidden_layers]. "
            "Note: must be set explicitly if custom values were used to launch vllm"
        ),
    )
    parser.add_argument(
        "--target-hidden-state-format",
        choices=["standard", "deepseek_v4_mean_hc_head"],
        default="standard",
        help="Opt-in DSV4 auxiliary-mean / post-hc_head teacher HS contract.",
    )
    parser.add_argument(
        "--dsv4-external-arrow",
        action="store_true",
        help=(
            "Accept external tokenized text Arrow without this repo's data manifest. "
            "You confirm its tokenizer/template/masks match the current "
            "DSV4 target. Checks all rows structurally and caps token/mask prefixes "
            "at --total-seq-len before HS requests without rewriting Arrow/order. "
            "Never bypasses an existing manifest or the target HS/checkpoint contract."
        ),
    )
    parser.add_argument(
        "--token-freq-path",
        type=str,
        default=None,
        help=(
            "Path to token frequency distribution file (.pt). Used together with "
            "--draft-vocab-size to build vocab mappings at training time. Falls back "
            "to '<data-path>/token_freq.pt' if not provided. If neither that file "
            "exists nor --draft-vocab-size is set, vocab mapping is skipped and the "
            "full verifier vocab is used."
        ),
    )
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=None,
        help=(
            "Vocabulary size for the draft model. Must be provided together with a "
            "token frequency file (--token-freq-path or '<data-path>/token_freq.pt') "
            "to generate vocab mappings. If either is absent, vocab mapping is skipped "
            "and the full verifier vocab is used, making this argument a no-op."
        ),
    )
    parser.add_argument("--d2t-path", type=str, default=None)
    parser.add_argument("--t2d-path", type=str, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--ttt-steps", type=int, default=3)
    parser.add_argument(
        "--num-speculative-steps",
        type=int,
        default=3,
        help="Number of MTP prediction steps (default: 3). Only used with MTP.",
    )
    parser.add_argument("--ttt-step-loss-decay", type=float, default=1.0)
    parser.add_argument(
        "--loss-fn",
        type=str,
        default="kl_div",
        help=(
            "Loss function specification. Pass a name for a single loss "
            "(kl_div, rkl, jsd, ce, tv, nla, lk_hybrid) or a JSON dict for a weighted "
            'combination, e.g. \'{"ce": 0.1, "tv": 0.9}\'. The DSpark default '
            "is the paper's CE=0.1, TV=0.9 combination."
        ),
    )
    parser.add_argument(
        "--step-weight-beta",
        type=float,
        default=0.6,
        help=(
            "Exponential decay factor for MTP step weights. "
            "Higher values weight earlier prediction steps more heavily. "
            "Only used with MTP algorithm."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--hidden-states-dtype",
        type=str,
        default="bfloat16",
        help="Data type for dataloader hidden states and autocast compute. "
        "Model master weights are always kept in fp32. "
        "Options: float32 (full precision), bfloat16 (recommended). "
        "Note: float16 is not supported (requires gradient scaling).",
    )
    parser.add_argument(
        "--deterministic-cuda",
        action="store_true",
        default=False,
        help="Sets cuda to deterministic mode. This may impact performance.",
    )
    # Model hyperparameters
    parser.add_argument(
        "--norm-before-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toggle normalization before residual connections (default: True)",
    )
    parser.add_argument(
        "--embed-requires-grad",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to train embedding layer weights (default: False)",
    )
    parser.add_argument(
        "--norm-before-fc",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply a single RMSNorm to the concatenated auxiliary hidden states "
        "before the FC projection (gpt-oss style). See --fc-norm for the "
        "per-layer alternative from the Eagle 3.1 paper. "
        "(default: True for eagle3, False otherwise). "
        "Disable with --no-norm-before-fc.",
    )
    parser.add_argument(
        "--fc-norm",
        action="store_true",
        default=False,
        help="Apply per-layer RMSNorm to each auxiliary hidden state before "
        "concatenation and FC projection (Eagle 3.1 paper approach).",
    )
    parser.add_argument(
        "--norm-output",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Feed post-norm hidden states back across TTT steps to stabilize "
        "magnitude drift across speculation depths "
        "(default: True for eagle3, False otherwise). "
        "Disable with --no-norm-output.",
    )
    # D-Flash specific parameters
    parser.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="Draft block size (default: 8; DSpark paper default: 7).",
    )
    parser.add_argument(
        "--sample-from-anchor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Sample from the anchor position (all positions predict). "
        "Default: False for dflash, True for dspark. ",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=3072,
        help="Maximum anchor positions for DFlash, DSpark, "
        "and P-EAGLE training (default: 3072).",
    )
    parser.add_argument(
        "--dflash-decay-gamma",
        type=float,
        default=4.0,
        help=(
            "Decay gamma for DFlash/DSpark loss weighting (default: 4.0; "
            "DSpark paper default: its block size)."
        ),
    )
    # D-Pace specific arguments (loss weight option + smoothing)
    parser.add_argument(
        "--per-position-loss-weight",
        choices=["fixed-exp-decay", "dpace"],
        default="fixed-exp-decay",
        help="Per-position loss weight option for D-PACE support"
        "default: fixed-exp-decay",
    )
    parser.add_argument(
        "--dpace-alpha",
        type=float,
        default=0.5,
        help="Smoothing constant for D-PACE loss (default: 0.5)",
    )
    add_muse_backbone_args(parser, muse_defaults)
    # DSpark baseline heads and Muse-specific Correction extensions.
    parser.add_argument(
        "--markov-rank",
        type=int,
        default=256,
        help="DSpark: low-rank dim of the Markov logit-bias head (0 disables it).",
    )
    parser.add_argument(
        "--markov-head-type",
        type=str,
        default="vanilla",
        choices=["vanilla", "gated", "rnn"],
        help="DSpark: sequential head variant (default: vanilla).",
    )
    add_muse_correction_args(parser, muse_defaults)
    parser.add_argument(
        "--enable-confidence-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DSpark: attach the per-position acceptance confidence head.",
    )
    parser.add_argument(
        "--confidence-head-with-markov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DSpark: feed the active sequential state into the confidence "
        "head alongside the backbone hidden state.",
    )
    parser.add_argument(
        "--confidence-detach-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DSpark: detach all ConfidenceHead inputs so its loss is auxiliary.",
    )
    parser.add_argument(
        "--confidence-head-alpha",
        type=float,
        default=1.0,
        help="DSpark: weight of the confidence-head BCE term (default: 1.0).",
    )
    parser.add_argument(
        "--confidence-length-alpha",
        type=float,
        default=0.0,
        help="DSpark: Smooth-L1 weight on predicted accept length (default: 0).",
    )
    parser.add_argument(
        "--confidence-loss-weighting",
        type=str,
        default="match-draft",
        choices=["uniform", "match-draft"],
        help="DSpark: how to weight confidence BCE over positions "
        "(default: match-draft, matching the DSpark objective).",
    )
    parser.add_argument(
        "--first-error-focal-alpha",
        type=float,
        default=0.0,
        help="DSpark: weight of first-error focal CE (default: 0).",
    )
    parser.add_argument(
        "--adaptive-loss",
        type=str,
        default="none",
        choices=["none", "cat", "ssal"],
        help="DSpark: adaptive position weights (default: none = fixed decay).",
    )
    parser.add_argument(
        "--ssal-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DSpark: mix decay→SSAL over training (requires --adaptive-loss ssal).",
    )
    parser.add_argument(
        "--ssal-curriculum-start",
        type=float,
        default=0.1,
        help="Progress fraction where decay→SSAL mix begins (default: 0.1).",
    )
    parser.add_argument(
        "--ssal-curriculum-end",
        type=float,
        default=0.6,
        help="Progress fraction where mix reaches pure SSAL (default: 0.6).",
    )
    parser.add_argument(
        "--draft-attn-impl",
        type=str,
        default="simple_flex_attention",
        choices=["simple_flex_attention", "sdpa", "eager"],
        help="Attention implementation for draft layers. "
        "Use 'sdpa' or 'eager' for hardware that doesn't support flex attention."
        "Not supported for MTP.",
    )
    # P-EAGLE specific parameters
    parser.add_argument(
        "--num-depths",
        type=int,
        default=8,
        help="Number of parallel prediction depths for P-EAGLE (default: 8)",
    )
    parser.add_argument(
        "--down-sample-ratio",
        type=float,
        default=0.7,
        help="Geometric decay ratio for COD sampling in P-EAGLE (default: 0.7)",
    )
    parser.add_argument(
        "--down-sample-ratio-min",
        type=float,
        default=0.2,
        help="Minimum retention ratio for COD sampling in P-EAGLE (default: 0.2)",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=2048,
        help="Sliding window size for sliding window attention layers (default: 2048). "
        "All draft layers use sliding window by default (except mtp).",
    )
    parser.add_argument(
        "--full-attention-indices",
        type=int,
        nargs="+",
        default=[],
        help="(Optional) Space-separated draft layer indices that should use full "
        "attention instead of sliding window. All draft layers use sliding window "
        "by default (except mtp). "
        "(e.g. '--full-attention-indices 0 2' makes layers 0 and 2 use full "
        "attention; the rest use sliding window).",
    )
    parser.add_argument(
        "--sliding-window-non-causal",
        action="store_true",
        default=False,
        help="Use non-causal (bidirectional) masking within draft blocks for sliding "
        "window attention layers. Full attention layers are always bidirectional. "
        "Note: vLLM currently doesn't support these models.",
    )
    # Dataloader parameters
    parser.add_argument(
        "--num-workers", type=int, default=12, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=4, help="Dataloader prefetch factor"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Standard deviation for noise augmentation",
    )
    # Checkpoint Parameters
    parser.add_argument(
        "--checkpoint-freq",
        type=_checkpoint_freq,
        default=1.0,
        help="Save a checkpoint every N epochs. Values < 1 enable sub-epoch "
        "checkpointing (e.g. 0.5 = every half epoch).",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        default=False,
        help="Pointing to checkpoint with lowest validation loss.",
    )

    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        default=False,
        help="Recompute DFlash/DSpark dense decoder layers during backward to "
        "save activation memory. Leaves correction/other heads and loss unchanged. "
        "Training only; independent of --fsdp-shard. Disabled by default.",
    )

    # distributed strategy
    parser.add_argument(
        "--fsdp-shard",
        action="store_true",
        default=False,
        help="Shard model parameters across GPUs with FSDP. By default, "
        "parameters are fully replicated (DDP-like). Enable this when the "
        "model does not fit in a single GPU's memory.",
    )

    # lr scheduler
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "none"],
    )
    parser.add_argument("--scheduler-warmup-steps", type=int, default=None)
    parser.add_argument(
        "--scheduler-warmup-ratio",
        type=float,
        default=None,
        help=(
            "Warmup as a fraction of total scheduler steps, in [0, 1]. Ignored "
            "(with a warning) when --scheduler-warmup-steps is also set."
        ),
    )
    parser.add_argument("--scheduler-total-steps", type=int, default=None)
    parser.add_argument("--scheduler-num-cosine-cycles", type=float, default=0.5)

    # optimizer
    parser.add_argument(
        "--optimizer",
        type=str,
        default="muon",
        choices=["adamw", "muon"],
        help=(
            "Optimizer to use. 'muon' applies Muon to 2D weight matrices and AdamW to "
            "the remaining params (norms, biases, embeddings, lm_head)."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for the AdamW optimizer (and the AdamW group in muon mode).",
    )
    parser.add_argument(
        "--muon-lr",
        type=float,
        default=None,
        help="LR for the Muon (2D weights) group. Only used with --optimizer muon. "
        "Defaults to 10*lr (and --lr defaults to 1e-4)",
    )
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn",
        type=str,
        default="match_rms_adamw",
        choices=["original", "match_rms_adamw"],
        help="Muon LR adjustment. 'match_rms_adamw' matches AdamW's update RMS.",
    )

    return parser


def _apply_training_defaults(args: argparse.Namespace, provided: set[str]) -> None:
    """Apply the selected algorithm defaults without replacing explicit options."""
    if args.speculator_type in ("dspark", "muse"):
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

    is_eagle3 = args.speculator_type == "eagle3"
    if args.draft_arch is None:
        args.draft_arch = "llama" if is_eagle3 else "qwen3"
    if args.norm_before_fc is None:
        args.norm_before_fc = is_eagle3
    if args.norm_output is None:
        args.norm_output = is_eagle3
    if args.muon_lr is None:
        args.muon_lr = 10 * args.lr


def _validate_training_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, provided: set[str]
) -> None:
    """Validate initialized arguments in the established CLI error order."""
    validate_draft_init_args(parser, args, provided)
    resolve_loss_config(args.loss_fn)

    try:
        if args.from_pretrained:
            # The remaining values will come from the checkpoint, not parser
            # defaults. Check only explicitly supplied scalars at this stage.
            validate_muse_options(
                {
                    field: getattr(args, field)
                    for field in args._provided_model_config_dests  # noqa: SLF001
                    & MUSE_MODEL_CONFIG_FIELDS
                },
                partial=True,
            )
        elif args.speculator_type == "muse":
            validate_muse_options(vars(args))
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
    if args.dsv4_external_arrow and (
        args.target_hidden_state_format != "deepseek_v4_mean_hc_head"
        or args.speculator_type not in ("dspark", "muse")
        or args.legacy_data
    ):
        parser.error(
            "--dsv4-external-arrow requires DSV4 DSpark/Muse training with Arrow data"
        )
    # This CLI-owned namespace metadata is shared with checkpoint reconciliation.
    args._provided_model_config_dests = explicitly_provided_dests(  # noqa: SLF001
        parser, PRETRAINED_MODEL_CONFIG_FLAGS, argv=argv
    )
    if (
        not args.from_pretrained
        and args.speculator_type != "muse"
        and (
            args._provided_model_config_dests  # noqa: SLF001
            & MUSE_MODEL_CONFIG_FIELDS
        )
    ):
        parser.error(
            "Correction, backbone enhancement and Selector options now belong to "
            "Muse; use --speculator-type muse. DFlash and DSpark select the "
            "baseline architectures."
        )

    # Preserve the shared CLI defaults for every other algorithm while making a
    # bare DSpark/Muse runs inherit the DSpark paper training recipe. Muse's
    # optional architecture extensions remain explicit opt-ins.
    dspark_default_dests = {
        "block_size",
        "dflash_decay_gamma",
        "epochs",
        "loss_fn",
        "num_layers",
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
