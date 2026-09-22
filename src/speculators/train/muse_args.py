"""Muse argument registration without model or training-runtime dependencies."""

import argparse
from collections.abc import Mapping


def add_muse_backbone_args(
    parser: argparse.ArgumentParser, muse_defaults: Mapping[str, object]
) -> None:
    """Register Muse backbone options using caller-provided defaults."""
    parser.add_argument(
        "--dflash-context-residual",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["dflash_context_residual"],
        help=(
            "Muse: inject the last inference-available verifier hidden "
            "state into each draft block (default: disabled)."
        ),
    )
    parser.add_argument(
        "--dflash-block-position-embedding",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["dflash_block_position_embedding"],
        help=(
            "Muse: add zero-initialized block-relative slot embeddings "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--dflash-gated-layer-fusion",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["dflash_gated_layer_fusion"],
        help=(
            "Muse: use normalized per-token gated auxiliary-layer fusion "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--dflash2-dynamic-conv",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["dflash2_dynamic_conv"],
        help=(
            "Muse: wrap every draft Attention and MLP with grouped causal "
            "dynamic convolutions (default: disabled)."
        ),
    )
    parser.add_argument(
        "--dflash2-conv-kernel-size",
        type=int,
        default=muse_defaults["dflash2_conv_kernel_size"],
        help="DFlash2 dynamic-convolution causal tap count (default: 2).",
    )
    parser.add_argument(
        "--dflash2-conv-group-size",
        type=int,
        default=muse_defaults["dflash2_conv_group_size"],
        help="DFlash2 hidden channels per dynamic-convolution group (default: 16).",
    )
    parser.add_argument(
        "--dflash2-candidate-selector",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["dflash2_candidate_selector"],
        help=(
            "Muse: sequentially re-rank the existing LM-head Top-K candidate "
            "chain (default: disabled)."
        ),
    )
    parser.add_argument(
        "--dflash2-selector-rank",
        type=int,
        default=muse_defaults["dflash2_selector_rank"],
        help="DFlash2 candidate-selector transition rank (default: 256).",
    )
    parser.add_argument(
        "--dflash2-selector-top-k",
        type=int,
        default=muse_defaults["dflash2_selector_top_k"],
        help="DFlash2 candidate-selector Top-K width (default: 16).",
    )
    selector_search_group = parser.add_mutually_exclusive_group()
    selector_search_group.add_argument(
        "--dflash2-selector-greedy",
        dest="dflash2_selector_search_mode",
        action="store_const",
        const="greedy",
        help="Use the original sequential greedy DFlash2 selector walk (default).",
    )
    selector_search_group.add_argument(
        "--dflash2-selector-global",
        dest="dflash2_selector_search_mode",
        action="store_const",
        const="global",
        help=(
            "Use global Viterbi search over locally normalized probabilities in "
            "the block-local Top-K selector lattice."
        ),
    )
    parser.set_defaults(
        dflash2_selector_search_mode=muse_defaults["dflash2_selector_search_mode"]
    )
    parser.add_argument(
        "--dflash2-selector-loss-weight",
        type=float,
        default=muse_defaults["dflash2_selector_loss_weight"],
        help="Weight of the DFlash2 restricted-Top-K selector loss (default: 1.0).",
    )


def add_muse_correction_args(
    parser: argparse.ArgumentParser, muse_defaults: Mapping[str, object]
) -> None:
    """Register Muse Correction options using caller-provided defaults."""
    parser.add_argument(
        "--enable-correction-head",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["enable_correction_head"],
        help=(
            "Muse: use causal Correction; it replaces Markov unless "
            "--correction-with-markov is enabled."
        ),
    )
    parser.add_argument(
        "--correction-output-mode",
        type=str,
        default=muse_defaults["correction_output_mode"],
        choices=["hidden", "logits"],
        help=(
            "Muse Correction output: 'hidden' adds a pre-LM-head hidden "
            "residual; 'logits' consumes previous logits and adds a low-rank "
            "vocabulary bias to DFlash base logits (default: hidden)."
        ),
    )
    parser.add_argument(
        "--correction-hidden-size",
        type=int,
        default=muse_defaults["correction_hidden_size"],
        help="Muse correction-head hidden width (default: 512).",
    )
    parser.add_argument(
        "--correction-rank",
        type=int,
        default=muse_defaults["correction_rank"],
        help="Muse correction residual bottleneck (default: 256).",
    )
    parser.add_argument(
        "--correction-lm-head-fusion",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["correction_lm_head_fusion"],
        help=(
            "Muse Correction: during no-grad rollout, project the DFlash block "
            "once and fuse low-rank hidden residuals with the LM head. Supports "
            "hidden mode and logits mode with corrected-hidden projection "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--correction-num-layers",
        type=int,
        default=muse_defaults["correction_num_layers"],
        help="Muse correction-head causal layers (default: 1).",
    )
    parser.add_argument(
        "--correction-num-heads",
        type=int,
        default=muse_defaults["correction_num_heads"],
        help="Muse correction-head attention heads (default: 8).",
    )
    parser.add_argument(
        "--correction-gate-bias",
        type=float,
        default=muse_defaults["correction_gate_bias"],
        help="Muse initial correction residual-gate bias (default: 0).",
    )
    parser.add_argument(
        "--correction-hidden-aux-loss",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["correction_hidden_aux_loss"],
        help=(
            "Muse: align Correction's corrected DFlash hidden with verifier "
            "pre-LM hidden using an auxiliary SmoothL1 loss (default: disabled)."
        ),
    )
    parser.add_argument(
        "--correction-hidden-aux-weight",
        type=float,
        default=muse_defaults["correction_hidden_aux_weight"],
        help="Muse hidden auxiliary-loss weight (default: 0.1).",
    )
    parser.add_argument(
        "--correction-hidden-feedback",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["correction_hidden_feedback"],
        help=(
            "Muse: feed each corrected hidden into the next Correction slot "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--selector-correction-feedback",
        choices=("static", "corrected"),
        default=muse_defaults["selector_correction_feedback"],
        help=(
            "Selector-to-Correction token feedback: static keeps the selected path; "
            "corrected feeds each Correction token into the next greedy slot."
        ),
    )
    parser.add_argument(
        "--correction-project-corrected-hidden",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["correction_project_corrected_hidden"],
        help=(
            "Muse logits mode: compute current logits as "
            "LMHead(h_DFlash + delta_hidden) + delta_logits while retaining one "
            "full LM-head projection (default: disabled)."
        ),
    )
    parser.add_argument(
        "--correction-with-markov",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["correction_with_markov"],
        help=(
            "Muse: jointly add a Correction-gated low-rank Markov bias after "
            "Correction's single LM-head projection (default: disabled)."
        ),
    )
    parser.add_argument(
        "--correction-markov-gate-bias",
        type=float,
        default=muse_defaults["correction_markov_gate_bias"],
        help="Muse initial collaboration gate bias (default: -2.0).",
    )
    parser.add_argument(
        "--correction-rollout-metrics",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["correction_rollout_metrics"],
        help=(
            "Muse Correction: measure greedy self-feedback metrics during "
            "validation (default: disabled for DSpark baseline parity)."
        ),
    )
    parser.add_argument(
        "--correction-base-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=muse_defaults["correction_base_diagnostics"],
        help="Muse: add a validation-only base projection for change/gain metrics.",
    )
