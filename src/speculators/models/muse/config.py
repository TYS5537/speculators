"""Configuration for the MUSE extensions to DFlash and DSpark."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.muse.backbone_config import MuseBackboneConfigMixin


class MuseOptions(MuseBackboneConfigMixin):
    """Shared opt-in architecture fields for config, factories and CLI validation."""

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


MUSE_OPTION_FIELDS = frozenset(MuseOptions.model_fields)


def muse_option_defaults() -> dict[str, Any]:
    """Return independent option defaults without constructing a model config."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in MuseOptions.model_fields.items()
    }


def validate_muse_options(values: Mapping[str, Any], *, partial: bool = False) -> None:
    """Validate scalar options and, when resolved, their architecture dependencies.

    A partial CLI overlay is not the checkpoint's architecture: only field-level
    ranges/choices can be checked before the saved configuration has been read.
    Decoder dimensions and vocabulary-dependent checks remain with their modules.
    """
    options = MuseOptions.model_validate(dict(values))
    if partial:
        return

    def context(name: str):
        default = DSparkSpeculatorConfig.model_fields[name].get_default(
            call_default_factory=True
        )
        value = values.get(name, default)
        if value is None and name in {
            "enable_confidence_head",
            "confidence_head_with_markov",
        }:
            return default
        return value

    correction = options.enable_correction_head
    markov_rank = context("markov_rank")
    markov_head_type = context("markov_head_type")
    _validate_selector_dependencies(options, markov_rank)
    _validate_correction_dependencies(options)
    if options.correction_with_markov:
        if not correction:
            raise ValueError("correction_with_markov=True requires Correction")
        if markov_rank <= 0:
            raise ValueError("correction_with_markov=True requires markov_rank > 0")
        if markov_head_type == "rnn":
            raise ValueError(
                "Correction-Markov collaboration supports only vanilla "
                "or gated Markov heads"
            )
    if (
        context("enable_confidence_head")
        and context("confidence_head_with_markov")
        and markov_rank <= 0
        and not correction
    ):
        raise ValueError(
            "confidence_head_with_markov=True requires an enabled Markov "
            "or correction head."
        )


def _validate_selector_dependencies(options: MuseOptions, markov_rank: int) -> None:
    if (
        options.dflash2_candidate_selector
        and options.dflash2_selector_search_mode == "global"
        and markov_rank > 0
        and not options.enable_correction_head
    ):
        raise ValueError(
            "Global DFlash2 path search must run before Correction and is not "
            "compatible with a standalone predecessor-dependent Markov head"
        )
    if options.selector_correction_feedback == "corrected":
        if not options.dflash2_candidate_selector or not options.enable_correction_head:
            raise ValueError(
                "selector_correction_feedback='corrected' requires Selector "
                "and Correction"
            )
        if options.dflash2_selector_search_mode != "greedy":
            raise ValueError(
                "selector_correction_feedback='corrected' requires greedy "
                "Selector search"
            )


def _validate_correction_dependencies(options: MuseOptions) -> None:
    correction = options.enable_correction_head
    if correction and options.correction_hidden_size % options.correction_num_heads:
        raise ValueError(
            "correction_hidden_size must be divisible by correction_num_heads"
        )
    if options.correction_output_mode != "hidden" and not correction:
        raise ValueError("correction_output_mode='logits' requires Correction")
    if options.correction_lm_head_fusion:
        if not correction:
            raise ValueError("correction_lm_head_fusion=True requires Correction")
        if (
            options.correction_output_mode == "logits"
            and not options.correction_project_corrected_hidden
        ):
            raise ValueError(
                "Logit Correction LM-head fusion requires "
                "correction_project_corrected_hidden=True"
            )
    if (
        options.correction_hidden_aux_loss
        or options.correction_hidden_feedback
        or options.correction_project_corrected_hidden
    ) and not correction:
        raise ValueError("Correction auxiliary/feedback features require Correction")
    if (
        options.correction_project_corrected_hidden
        and options.correction_output_mode != "logits"
    ):
        raise ValueError("correction_project_corrected_hidden requires logits mode")


@SpeculatorModelConfig.register("muse")
class MuseSpeculatorConfig(MuseOptions, DSparkSpeculatorConfig):
    """Optional backbone, Selector, Correction, and collaboration architecture.

    Existing flag and tensor names intentionally stay unchanged. Selecting MUSE
    does not implicitly enable an experiment; use the explicit feature switches.
    """

    speculators_model_type: Literal["muse"] = "muse"  # type: ignore[assignment]
    architectures: list[str] = Field(default_factory=lambda: ["MuseSpeculator"])
