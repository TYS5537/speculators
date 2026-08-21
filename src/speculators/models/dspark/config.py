from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DSparkSpeculatorConfig",
]


@SpeculatorModelConfig.register("dspark")
class DSparkSpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash config plus sequential correction and confidence heads.

    The active sequential head lets each draft position condition on previously
    sampled tokens within the block; the confidence head predicts per-position
    acceptance probability. All DFlash fields are inherited unchanged.
    """

    speculators_model_type: Literal["dspark"] = "dspark"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DSparkSpeculator"],
        description="Model architectures that can load these weights",
    )

    block_size: int = Field(
        default=7,
        description="DSpark paper proposal length (gamma).",
    )

    sample_from_anchor: bool = Field(
        default=True,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
            "Default True matches DeepSeek/DeepSpec convention."
        ),
    )

    # Sequential (Markov) head.
    markov_rank: int = Field(
        default=256,
        description=(
            "Low-rank dimension of the Markov logit-bias factorization B = W1 @ W2. "
            "Set to 0 to disable the sequential head (pure DFlash drafting)."
        ),
    )
    markov_head_type: Literal["vanilla", "gated", "rnn"] = Field(
        default="vanilla",
        description=(
            "Sequential head variant: 'vanilla' (first-order Markov bias), 'gated' "
            "(hidden-gated bias), or 'rnn' (recurrent state over the block)."
        ),
    )

    # Causal correction head. By default it replaces MarkovHead.
    enable_correction_head: bool = Field(
        default=False,
        description=(
            "Replace the Markov head with a causal head. Hidden mode predicts a "
            "pre-projection hidden residual; logits mode consumes previous target/"
            "generated logits and predicts a low-rank vocabulary bias."
        ),
    )
    correction_output_mode: Literal["hidden", "logits"] = Field(
        default="hidden",
        description=(
            "Correction output space. 'hidden' preserves the single full LM-head "
            "baseline. 'logits' adds a Markov-like low-rank bias to DFlash base "
            "logits and feeds the previous position's logits into Correction."
        ),
    )
    correction_hidden_size: int = Field(
        default=512,
        gt=0,
        description="Hidden width of the causal correction head.",
    )
    correction_rank: int = Field(
        default=256,
        gt=0,
        description="Low-rank bottleneck used to produce the correction residual.",
    )
    correction_lm_head_fusion: bool = Field(
        default=False,
        description=(
            "During no-grad rollout, compute the block base logits once and fuse "
            "Correction's low-rank hidden residual with the LM head. This supports "
            "hidden output and logit output with corrected-hidden projection. "
            "Training and the default disabled path are unchanged."
        ),
    )
    correction_num_layers: int = Field(
        default=1,
        gt=0,
        description="Number of causal Transformer layers in the correction head.",
    )
    correction_num_heads: int = Field(
        default=8,
        gt=0,
        description="Attention heads in each correction layer.",
    )
    correction_gate_bias: float = Field(
        default=0.0,
        description="Initial bias of the sigmoid correction-residual gate.",
    )
    correction_hidden_aux_loss: bool = Field(
        default=False,
        description=(
            "Add an auxiliary SmoothL1 objective that aligns Correction's "
            "corrected DFlash hidden state with the aligned verifier pre-LM hidden."
        ),
    )
    correction_hidden_aux_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight of the optional Correction hidden-alignment loss.",
    )
    correction_hidden_feedback: bool = Field(
        default=False,
        description=(
            "Feed each corrected DFlash hidden state into the next correction slot. "
            "This makes teacher-forced Correction sequential and is disabled by "
            "default for baseline parity."
        ),
    )
    selector_correction_feedback: Literal["static", "corrected"] = Field(
        default="static",
        description=(
            "How an upstream DFlash2 Selector conditions sequential Correction. "
            "'static' keeps the preselected path fixed; 'corrected' feeds each "
            "Correction output token into the next greedy Selector/Correction slot."
        ),
    )
    correction_project_corrected_hidden: bool = Field(
        default=False,
        description=(
            "In logits mode, project h_DFlash + delta_hidden through the sole "
            "LM head before adding delta_logits. Disabled preserves the parallel "
            "auxiliary-hidden baseline."
        ),
    )
    correction_with_markov: bool = Field(
        default=False,
        description=(
            "Jointly apply the low-rank Markov logit bias after Correction's single "
            "full-vocabulary projection. Its global residual scale starts at zero; "
            "the feature supports vanilla/gated Markov heads and is disabled by "
            "default for baseline parity."
        ),
    )
    correction_markov_gate_bias: float = Field(
        default=-2.0,
        description=(
            "Initial bias of the Correction-state gate controlling the collaborative "
            "Markov logit bias."
        ),
    )
    correction_rollout_metrics: bool = Field(
        default=False,
        description=(
            "Measure greedy self-feedback correction metrics during validation. "
            "Disabled by default because it is not part of the DSpark baseline."
        ),
    )
    correction_base_diagnostics: bool = Field(
        default=False,
        description=(
            "During validation only, run an extra base LM-head projection for "
            "change/gain diagnostics when hidden mode is active. Logits mode already "
            "has base logits and does not need the extra projection."
        ),
    )

    # Confidence head.
    enable_confidence_head: bool = Field(
        default=True,
        description="Whether to attach the per-position acceptance-probability head.",
    )
    confidence_head_with_markov: bool = Field(
        default=True,
        description=(
            "Concatenate the active sequential state (Markov embedding or causal "
            "correction state) with the backbone hidden state for confidence."
        ),
    )
    confidence_detach_features: bool = Field(
        default=False,
        description=(
            "Detach backbone and sequential features before ConfidenceHead. "
            "False lets confidence loss train the active Markov/correction path; "
            "True makes confidence a fully auxiliary observer."
        ),
    )
