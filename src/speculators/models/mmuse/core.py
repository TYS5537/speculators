from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.model_definitions import (
    ConfidenceHead,
    MarkovHead,
)
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.mmuse.backbone import MMuseBackboneMixin
from speculators.models.mmuse.config import (
    MMuseSpeculatorConfig,
    validate_mmuse_options,
)
from speculators.models.mmuse.correction import CausalCorrectionHead, CorrectionCache
from speculators.models.mmuse.feedback_correction import build_feedback_conditioning
from speculators.models.mmuse.metrics import compute_metrics, select_logged_metrics
from speculators.models.mmuse.parallel_correction import (
    add_parallel_base_residual,
    add_parallel_hidden_residual,
    add_parallel_projected_residual,
    build_parallel_logit_kwargs,
)
from speculators.models.mmuse.rollout_state import RolloutFeedbackState
from speculators.models.mmuse.rollout_validation import (
    validate_selector_logit_features,
    validate_selector_token_ids,
)
from speculators.models.mmuse.runtime_types import (
    CorrectionRolloutOutput,
    CorrectionStepOutput,
    InitialLogitFeedback,
    SelectorConditioning,
    SelectorCorrectionInputs,
    TeacherForcedCorrectionOutput,
    TrainingBlocks,
)
from speculators.models.mmuse.selector_inputs import (
    align_selector_token_ids,
    shift_selector_candidates,
)
from speculators.models.mmuse.training_inputs import prepare_training_blocks
from speculators.models.utils import conditional_torch_compile

_DSPARK_PAPER_LOSS_FN = '{"ce": 0.1, "tv": 0.9}'
_DEFAULT_LOSS_CONFIG: LossConfig = resolve_loss_config(_DSPARK_PAPER_LOSS_FN)
_BLOCK_HIDDEN_RANK = 3

__all__ = [
    "MMuseDraftModel",
]


@SpeculatorModel.register("mmuse")
class MMuseDraftModel(MMuseBackboneMixin, DSparkDraftModel):
    """MMUSE: enhanced DFlash backbone with causal Correction and confidence.

    The legacy Markov path refines base logits. The causal Correction path can
    either refine DFlash hidden states before the sole LM-head projection or
    consume previous logits and refine base logits with a low-rank vocabulary
    bias. An opt-in collaboration path gates a further Markov bias from Correction
    state. Optional hidden alignment and corrected-hidden feedback provide
    representation-level supervision and recurrence. The confidence head predicts
    each position's acceptance probability.
    """

    config_class: ClassVar[type[MMuseSpeculatorConfig]] = MMuseSpeculatorConfig  # type: ignore[misc,assignment]

    def _init_sequential_heads(self, config: MMuseSpeculatorConfig) -> None:
        validate_mmuse_options(vars(config))
        hidden_size = config.transformer_layer_config.hidden_size
        self.markov_head: MarkovHead | None = None
        self.correction_head: CausalCorrectionHead | None = None
        self.correction_markov_gate: torch.nn.Linear | None = None
        self.correction_markov_scale: torch.nn.Parameter | None = None
        if config.enable_correction_head:
            self.correction_head = CausalCorrectionHead(
                input_hidden_size=hidden_size,
                token_embedding_size=hidden_size,
                block_size=self.block_size,
                correction_hidden_size=config.correction_hidden_size,
                correction_rank=config.correction_rank,
                num_layers=config.correction_num_layers,
                num_heads=config.correction_num_heads,
                gate_bias=config.correction_gate_bias,
                output_mode=config.correction_output_mode,
                draft_vocab_size=self.draft_vocab_size,
                enable_hidden_auxiliary=(
                    config.correction_hidden_aux_loss
                    or config.correction_project_corrected_hidden
                ),
                enable_hidden_feedback=config.correction_hidden_feedback,
            )
            if config.correction_with_markov:
                self.markov_head = MarkovHead(
                    verifier_vocab_size=self.verifier_vocab_size,
                    draft_vocab_size=self.draft_vocab_size,
                    markov_rank=config.markov_rank,
                    hidden_size=hidden_size,
                    head_type=config.markov_head_type,
                )
                self.correction_markov_gate = torch.nn.Linear(
                    config.correction_hidden_size, 1
                )
                self.correction_markov_scale = torch.nn.Parameter(torch.zeros(()))
                torch.nn.init.zeros_(self.correction_markov_gate.weight)
                torch.nn.init.constant_(
                    self.correction_markov_gate.bias,
                    config.correction_markov_gate_bias,
                )
        elif config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            sequential_dim = 0
            if config.confidence_head_with_markov:
                sequential_dim = (
                    config.correction_hidden_size
                    if self.correction_head is not None
                    else config.markov_rank
                )
            input_dim = hidden_size + sequential_dim
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "MMuseDraftModel":
        """Create the enhanced MMUSE model without changing parameter names."""
        fields = cls.config_class.model_fields
        kwargs.setdefault("block_size", fields["block_size"].get_default())
        sequential_kwargs = {}
        for name in (
            "markov_rank",
            "markov_head_type",
            "enable_confidence_head",
            "confidence_head_with_markov",
            "confidence_detach_features",
        ):
            default = fields[name].get_default()
            value = kwargs.get(name, default)
            if value is None and name in {
                "enable_confidence_head",
                "confidence_head_with_markov",
            }:
                value = default
            sequential_kwargs[name] = value
        config = cls.config_class(
            **cls._build_base_config_kwargs("mmuse", verifier_config, **kwargs),
            **sequential_kwargs,
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _hidden_alignment_loss(
        corrected_hidden: torch.Tensor,
        verifier_hidden: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked SmoothL1 alignment in the shared pre-LM hidden space."""
        if corrected_hidden.shape != verifier_hidden.shape:
            raise ValueError("Corrected and verifier hidden states must align")
        if loss_mask.shape != corrected_hidden.shape[:-1]:
            raise ValueError("Hidden-alignment mask must match token dimensions")
        per_token = torch.nn.functional.smooth_l1_loss(
            corrected_hidden.float(),
            verifier_hidden.float(),
            reduction="none",
        ).mean(dim=-1)
        mask = loss_mask.to(per_token.dtype)
        return (per_token * mask).sum() / mask.sum().clamp_min(1.0)

    def _teacher_forced_parallel_correction(
        self,
        hidden_blocks: torch.Tensor,
        correction_previous_ids: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        targets: torch.Tensor,
        base_logits_blocks: torch.Tensor | None,
        selector_current_embeddings: torch.Tensor | None = None,
        selector_previous_rank_features: torch.Tensor | None = None,
        selector_previous_logits_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        """Correct active teacher-forced slots without sequential hidden feedback.

        Return flattened logits (or ``None`` in hidden mode), full-block causal
        states and optional corrected hidden states. A reserved anchor preserves
        its base hidden/logits and receives zero causal state. The caller retains
        the single final LM-head projection for hidden-output Correction.
        """
        if self.correction_head is None:
            raise RuntimeError("Parallel Correction requires an enabled head")
        num_blocks, block = hidden_blocks.shape[:2]
        mask_tokens_size = num_blocks * block
        start_position = 0 if self.config.sample_from_anchor else 1
        active_hidden = (
            hidden_blocks if start_position == 0 else hidden_blocks[:, start_position:]
        )
        active_positions = block_positions[:, start_position:]
        correction_output_mode = getattr(self.correction_head, "output_mode", "hidden")
        with torch.no_grad():
            previous_embeddings = self.embed_tokens(
                correction_previous_ids[:, start_position:]
            )
        correction_kwargs: dict[str, torch.Tensor | None] = {}
        if selector_current_embeddings is not None:
            correction_kwargs["current_token_embeddings"] = selector_current_embeddings[
                :, start_position:
            ]
        if correction_output_mode == "logits":
            if (
                base_logits_blocks is None
                and not self.config.correction_project_corrected_hidden
            ):
                raise RuntimeError("Logit-residual Correction requires base logits")
            correction_kwargs.update(
                build_parallel_logit_kwargs(
                    targets,
                    block_positions,
                    active_positions,
                    num_blocks=num_blocks,
                    block_size=block,
                    start_position=start_position,
                    selector_previous_rank_features=selector_previous_rank_features,
                    selector_previous_logits_mask=selector_previous_logits_mask,
                )
            )

        residual, draft_states, _ = self.correction_head(
            previous_embeddings,
            active_hidden,
            active_positions,
            **correction_kwargs,
        )
        delta_hidden = None
        if correction_output_mode != "logits":
            delta_hidden = residual
        elif (
            self.config.correction_project_corrected_hidden
            or self.config.correction_hidden_aux_loss
        ):
            delta_hidden = self.correction_head.auxiliary_hidden_residual(draft_states)
        corrected_hidden = add_parallel_hidden_residual(
            hidden_blocks, delta_hidden, start_position=start_position
        )

        logits = None
        if correction_output_mode == "logits":
            if self.config.correction_project_corrected_hidden:
                projected_logits = self.lm_head(
                    corrected_hidden.reshape(1, mask_tokens_size, -1).to(
                        self.lm_head.weight.dtype
                    )
                )
                logits = add_parallel_projected_residual(
                    projected_logits,
                    residual,
                    num_blocks=num_blocks,
                    start_position=start_position,
                    mask_tokens_size=mask_tokens_size,
                )
            else:
                assert base_logits_blocks is not None  # noqa: S101
                logits = add_parallel_base_residual(
                    base_logits_blocks,
                    residual,
                    start_position=start_position,
                    mask_tokens_size=mask_tokens_size,
                )
        correction_states = draft_states
        if start_position:
            correction_states = torch.cat(
                [
                    draft_states.new_zeros(
                        num_blocks, start_position, draft_states.shape[-1]
                    ),
                    draft_states,
                ],
                dim=1,
            )
        return logits, correction_states, corrected_hidden

    def _apply_collaborative_markov(
        self,
        correction_logits: torch.Tensor,
        correction_states: torch.Tensor,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate a low-rank Markov bias with Correction's causal state."""
        if (
            self.markov_head is None
            or self.correction_markov_gate is None
            or self.correction_markov_scale is None
        ):
            raise RuntimeError("Correction-Markov collaboration is not enabled")
        prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
        markov_bias = self.markov_head.block_bias(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
            prev_emb=prev_emb,
        )
        gate_dtype = self.correction_markov_gate.weight.dtype
        local_gate = torch.sigmoid(
            self.correction_markov_gate(correction_states.to(gate_dtype))
        )
        gate = torch.tanh(self.correction_markov_scale) * local_gate
        collaborative_logits = correction_logits + (
            gate.to(markov_bias.dtype) * markov_bias
        )
        return collaborative_logits, gate, prev_emb

    @torch.compiler.disable
    def _teacher_forced_hidden_feedback_correction(
        self,
        dflash_hidden: torch.Tensor,
        previous_token_embeddings: torch.Tensor,
        block_positions: torch.Tensor,
        base_logits: torch.Tensor | None,
        previous_target_logits: torch.Tensor | None,
        previous_target_logits_mask: torch.Tensor | None,
        previous_rank_features: torch.Tensor | None = None,
        current_token_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run teacher-forced Correction with differentiable hidden/cache feedback.

        The reserved anchor seeds hidden feedback without a head call or cache
        entry. Active slots prepare conditioning separately; final full-block
        projection happens only after every recurrent step has completed.
        """
        if self.correction_head is None:
            raise RuntimeError("Hidden feedback requires Correction")
        if not self.config.correction_hidden_feedback:
            raise RuntimeError("Correction hidden feedback is not enabled")

        num_blocks, block_size, hidden_size = dflash_hidden.shape
        start_position = 0 if self.config.sample_from_anchor else 1
        output_states: list[torch.Tensor] = []
        output_corrected_hidden: list[torch.Tensor] = []
        output_delta_logits: list[torch.Tensor] = []
        previous_corrected_hidden = dflash_hidden.new_zeros(num_blocks, 1, hidden_size)
        previous_corrected_hidden_mask = torch.zeros(
            num_blocks,
            1,
            dtype=torch.bool,
            device=dflash_hidden.device,
        )
        cache = None

        for position in range(block_size):
            current_hidden = dflash_hidden[:, position]
            if position < start_position:
                corrected_current_hidden = current_hidden
                causal_states = current_hidden.new_zeros(
                    num_blocks, self.config.correction_hidden_size
                )
                if self.correction_head.output_mode == "logits":
                    delta_logits = dflash_hidden.new_zeros(
                        num_blocks,
                        self.draft_vocab_size,
                        dtype=self.lm_head.weight.dtype,
                    )
            else:
                head_kwargs = {
                    "previous_corrected_hidden": previous_corrected_hidden,
                    "previous_corrected_hidden_mask": (previous_corrected_hidden_mask),
                    "cache": cache,
                    "use_cache": True,
                }
                head_kwargs.update(
                    build_feedback_conditioning(
                        position,
                        output_mode=self.correction_head.output_mode,
                        current_token_embeddings=current_token_embeddings,
                        previous_target_logits=previous_target_logits,
                        previous_target_logits_mask=previous_target_logits_mask,
                        previous_rank_features=previous_rank_features,
                    )
                )
                if self.correction_head.output_mode == "logits":
                    delta_logits_step, causal_step, cache = self.correction_head(
                        previous_token_embeddings[:, position : position + 1],
                        dflash_hidden[:, position : position + 1],
                        block_positions[:, position : position + 1],
                        **head_kwargs,
                    )
                    delta_logits = delta_logits_step[:, 0]
                    delta_hidden = self.correction_head.auxiliary_hidden_residual(
                        causal_step,
                    )
                else:
                    delta_hidden, causal_step, cache = self.correction_head(
                        previous_token_embeddings[:, position : position + 1],
                        dflash_hidden[:, position : position + 1],
                        block_positions[:, position : position + 1],
                        **head_kwargs,
                    )
                causal_states = causal_step[:, 0]
                corrected_current_hidden = current_hidden + delta_hidden[:, 0].to(
                    current_hidden.dtype
                )

            output_states.append(causal_states)
            output_corrected_hidden.append(corrected_current_hidden)
            if self.correction_head.output_mode == "logits":
                output_delta_logits.append(delta_logits)
            previous_corrected_hidden = corrected_current_hidden.unsqueeze(1)
            previous_corrected_hidden_mask = torch.ones(
                num_blocks,
                1,
                dtype=torch.bool,
                device=dflash_hidden.device,
            )

        corrected_hidden = torch.stack(output_corrected_hidden, dim=1)
        correction_states = torch.stack(output_states, dim=1)
        logits = self._project_hidden_feedback_logits(
            corrected_hidden,
            output_delta_logits,
            base_logits,
            num_blocks=num_blocks,
            block_size=block_size,
            hidden_size=hidden_size,
        )
        return logits, correction_states, corrected_hidden

    def _project_hidden_feedback_logits(
        self,
        corrected_hidden: torch.Tensor,
        output_delta_logits: list[torch.Tensor],
        base_logits: torch.Tensor | None,
        *,
        num_blocks: int,
        block_size: int,
        hidden_size: int,
    ) -> torch.Tensor:
        """Finish a recurrent block, preserving projection/cast and error ordering.

        Hidden and dual modes project once after the loop. Base-logit mode never
        projects here, and checks for missing base logits only after recurrence.
        The caller has already checked that Correction is enabled.
        """
        if self.correction_head.output_mode == "logits":
            delta_logits = torch.stack(output_delta_logits, dim=1)
            if self.config.correction_project_corrected_hidden:
                projected_logits = self.lm_head(
                    corrected_hidden.reshape(
                        1, num_blocks * block_size, hidden_size
                    ).to(self.lm_head.weight.dtype)
                ).view(num_blocks, block_size, -1)
                logits = projected_logits + delta_logits.to(projected_logits.dtype)
            else:
                if base_logits is None:
                    raise RuntimeError("Logit-residual Correction requires base logits")
                logits = base_logits + delta_logits.to(base_logits.dtype)
        else:
            logits = self.lm_head(
                corrected_hidden.reshape(1, num_blocks * block_size, -1).to(
                    self.lm_head.weight.dtype
                )
            ).view(num_blocks, block_size, -1)
        return logits

    @staticmethod
    def _replace_compact_feature_position(
        features: torch.Tensor | None,
        replacement: torch.Tensor | None,
        position: int,
    ) -> torch.Tensor | None:
        """Replace one compact sequence position without a vocabulary tensor."""
        if features is None:
            return None
        if replacement is None:
            raise RuntimeError("Replacement compact logit features are missing")
        return torch.cat(
            [features[:, :position], replacement, features[:, position + 1 :]],
            dim=1,
        )

    @staticmethod
    def _prepend_zero_compact_feature(
        features: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Prepend one masked zero position to a compact feature sequence."""
        if features is None:
            return None
        zero = features.new_zeros(features.shape[0], 1, features.shape[-1])
        return torch.cat([zero, features], dim=1)

    def _prepare_selector_conditioning(
        self,
        base_logits: torch.Tensor | None,
        targets: torch.Tensor,
        hidden_blocks: torch.Tensor,
        *,
        block_tokens: torch.Tensor,
        aligned_loss_mask: torch.Tensor,
        prev_token_ids: torch.Tensor,
        block_positions: torch.Tensor,
        correction_output_mode: str | None,
    ) -> SelectorConditioning:
        """Prepare teacher-forced Selector inputs only for joint Correction.

        Token embeddings remain frozen while compact previous-distribution
        features retain their gradient path. Standalone Selector handling and
        autoregressive corrected-token feedback are owned by their callers.

        The result names the loss, IDs, embeddings and compact logit features;
        its tuple order remains compatible with the original helper contract.
        """
        if self.candidate_selector is None or self.correction_head is None:
            return SelectorConditioning(
                selector_loss=None,
                previous_token_ids=prev_token_ids,
                current_token_embeddings=None,
                previous_rank_features=None,
                previous_logits_mask=None,
            )
        if base_logits is None:
            raise RuntimeError(
                "Selector-conditioned Correction requires pure DFlash base logits"
            )
        num_blocks, block = hidden_blocks.shape[:2]
        selector_previous_rank_features = None
        selector_previous_logits_mask = None
        (
            selector_candidate_ids,
            selector_candidate_logits,
            selector_loss,
            selector_selected_ids,
            selector_teacher_logits,
        ) = self._dflash2_block_outputs(
            base_logits,
            targets,
            hidden_blocks,
            block_tokens[:, 0],
            aligned_loss_mask,
            teacher_previous_token_ids=prev_token_ids,
        )
        selector_initial_logits = None
        if not self.config.sample_from_anchor:
            selector_initial_logits = targets.view(num_blocks, block, -1)[:, 0]
        if self.config.selector_correction_feedback == "corrected":
            selected_indices = selector_teacher_logits.argmax(dim=-1, keepdim=True)
            teacher_selected_ids = selector_candidate_ids.gather(
                -1, selected_indices
            ).squeeze(-1)
            # Main train/validation metrics remain teacher forced. Actual
            # corrected-token feedback is measured by rollout/offline eval.
            selector_current_ids = self._draft_ids_to_verifier(teacher_selected_ids)
            selector_previous_ids = prev_token_ids
            if correction_output_mode == "logits":
                teacher_source_mask = torch.ones(
                    num_blocks,
                    max(block - 1, 0),
                    dtype=torch.bool,
                    device=hidden_blocks.device,
                )
                teacher_rank = self.correction_head.encode_previous_distribution(
                    teacher_source_mask,
                    previous_logits=targets.view(num_blocks, block, -1)[:, :-1],
                )

                selector_previous_rank_features = self._prepend_zero_compact_feature(
                    teacher_rank
                )
                selector_previous_logits_mask = block_positions > 0
        else:
            selector_inputs = self._selector_correction_inputs(
                selector_candidate_ids,
                selector_candidate_logits,
                selector_selected_ids,
                block_tokens[:, 0],
                initial_previous_logits=selector_initial_logits,
            )
            selector_current_ids = selector_inputs.current_token_ids
            selector_previous_ids = selector_inputs.previous_token_ids
            selector_previous_rank_features = selector_inputs.previous_rank_features
            selector_previous_logits_mask = selector_inputs.previous_logits_mask
        selector_current_embeddings = None
        if selector_current_ids is not None:
            with torch.no_grad():
                selector_current_embeddings = self.embed_tokens(selector_current_ids)
        correction_previous_ids = (
            selector_previous_ids
            if selector_previous_ids is not None
            else prev_token_ids
        )
        return SelectorConditioning(
            selector_loss=selector_loss,
            previous_token_ids=correction_previous_ids,
            current_token_embeddings=selector_current_embeddings,
            previous_rank_features=selector_previous_rank_features,
            previous_logits_mask=selector_previous_logits_mask,
        )

    def _selector_correction_inputs(
        self,
        candidate_ids: torch.Tensor,
        selector_logits: torch.Tensor,
        selected_draft_ids: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        initial_previous_logits: torch.Tensor | None = None,
    ) -> SelectorCorrectionInputs:
        """Validate a static path, align its inputs, then encode compact features."""
        if self.correction_head is None:
            raise RuntimeError("Selector conditioning requires Correction")
        if candidate_ids.shape != selector_logits.shape:
            raise ValueError("Selector candidate IDs and logits must align")
        if selected_draft_ids.shape != candidate_ids.shape[:-1]:
            raise ValueError("Selector path must align with candidate block positions")
        if anchor_token_ids.shape != (candidate_ids.shape[0],):
            raise ValueError("Selector conditioning requires one anchor per block")

        # Greedy keeps its realized Top-K row. Global search is a deterministic
        # whole-path proposal, so its actual per-position q is the selected token's
        # one-hot row rather than the local edge argmax distribution.
        selector_logits = self._dflash2_proposal_logits(
            candidate_ids,
            selector_logits,
            selected_draft_ids,
        ).detach()

        num_blocks, block_size, _ = candidate_ids.shape
        start_position = 0 if self.config.sample_from_anchor else 1
        token_inputs = align_selector_token_ids(
            selected_draft_ids,
            anchor_token_ids,
            block_size=block_size,
            start_position=start_position,
            draft_to_verifier=self._draft_ids_to_verifier,
        )

        needs_logit_features = (
            getattr(self.correction_head, "output_mode", "hidden") == "logits"
        )
        if not needs_logit_features:
            return token_inputs

        previous_candidates = shift_selector_candidates(
            candidate_ids,
            selector_logits,
            sample_from_anchor=self.config.sample_from_anchor,
            initial_previous_logits=initial_previous_logits,
        )
        previous_rank = self.correction_head.encode_previous_distribution(
            previous_candidates.mask,
            candidate_ids=previous_candidates.candidate_ids,
            candidate_logits=previous_candidates.candidate_logits,
        )
        previous_logits_mask = previous_candidates.mask
        if not self.config.sample_from_anchor:
            if initial_previous_logits is None:
                raise ValueError(
                    "Logit-aware Selector Correction with sample_from_anchor=False "
                    "requires initial verifier logits"
                )
            expected = (num_blocks, self.draft_vocab_size)
            if initial_previous_logits.shape != expected:
                raise ValueError(
                    "Expected selector initial logits shape "
                    f"{expected}, got {tuple(initial_previous_logits.shape)}"
                )
            initial_mask = torch.ones(
                num_blocks,
                1,
                dtype=torch.bool,
                device=candidate_ids.device,
            )
            initial_rank = self.correction_head.encode_previous_distribution(
                initial_mask,
                previous_logits=initial_previous_logits.unsqueeze(1),
            )

            previous_rank = self._replace_compact_feature_position(
                previous_rank,
                initial_rank,
                start_position,
            )
            previous_logits_mask = previous_candidates.mask.clone()
            previous_logits_mask[:, start_position] = True
        return SelectorCorrectionInputs(
            current_token_ids=token_inputs.current_token_ids,
            previous_token_ids=token_inputs.previous_token_ids,
            previous_rank_features=previous_rank,
            previous_logits_mask=previous_logits_mask,
        )

    def _validate_rollout_inputs(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        precomputed_base_logits: torch.Tensor | None = None,
        conditioning_current_ids: torch.Tensor | None = None,
        conditioning_previous_ids: torch.Tensor | None = None,
        conditioning_previous_rank_features: torch.Tensor | None = None,
        conditioning_previous_logits_mask: torch.Tensor | None = None,
    ) -> tuple[bool, bool]:
        """Validate block, Selector IDs/features, then base logits in that order."""
        if self.correction_head is None:
            raise RuntimeError(
                "_rollout_correction_steps requires enable_correction_head=True"
            )
        if dflash_hidden.ndim != _BLOCK_HIDDEN_RANK:
            raise ValueError("dflash_hidden must be rank-3")
        if dflash_hidden.shape[1] != self.block_size:
            raise ValueError(
                f"Expected block_size={self.block_size}, got {dflash_hidden.shape[1]}"
            )
        if anchor_token_ids.shape != (dflash_hidden.shape[0],):
            raise ValueError(
                f"Expected anchor_token_ids shape {(dflash_hidden.shape[0],)}, "
                f"got {anchor_token_ids.shape}"
            )
        expected_block_shape = (dflash_hidden.shape[0], self.block_size)
        selector_feedback = getattr(
            self.config,
            "selector_correction_feedback",
            "static",
        )
        online_selector = (
            self.candidate_selector is not None and selector_feedback == "corrected"
        )
        validate_selector_token_ids(
            conditioning_current_ids,
            conditioning_previous_ids,
            expected_block_shape=expected_block_shape,
            online_selector=online_selector,
        )
        has_conditioning_features = validate_selector_logit_features(
            conditioning_previous_rank_features,
            conditioning_previous_logits_mask,
            expected_block_shape=expected_block_shape,
        )
        if precomputed_base_logits is not None:
            expected_base_shape = (
                dflash_hidden.shape[0],
                self.block_size,
                self.draft_vocab_size,
            )
            if precomputed_base_logits.shape != expected_base_shape:
                raise ValueError(
                    "Expected precomputed_base_logits shape "
                    f"{expected_base_shape}, got "
                    f"{tuple(precomputed_base_logits.shape)}"
                )
        if online_selector and precomputed_base_logits is None:
            raise ValueError("Corrected Selector feedback requires DFlash base logits")
        return online_selector, has_conditioning_features

    def _initial_rollout_logit_feedback(
        self,
        dflash_hidden: torch.Tensor,
        initial_previous_logits: torch.Tensor | None,
        *,
        logit_feedback_enabled: bool,
        has_conditioning_features: bool,
        online_selector: bool,
    ) -> InitialLogitFeedback:
        """Prepare dense or online compact logit feedback before the first slot."""
        previous_feedback_logits = None
        previous_feedback_mask = None
        online_previous_rank = None
        online_previous_mask = None
        if (
            logit_feedback_enabled
            and not has_conditioning_features
            and not online_selector
        ):
            draft_vocab_size = getattr(
                self, "draft_vocab_size", self.lm_head.out_features
            )
            expected_initial_shape = (
                dflash_hidden.shape[0],
                draft_vocab_size,
            )
            if self.config.sample_from_anchor:
                if initial_previous_logits is not None:
                    raise ValueError(
                        "initial_previous_logits is only valid when "
                        "sample_from_anchor=False"
                    )
                previous_feedback_logits = dflash_hidden.new_zeros(
                    dflash_hidden.shape[0],
                    1,
                    draft_vocab_size,
                    dtype=self.lm_head.weight.dtype,
                )
                previous_feedback_mask = torch.zeros(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
            else:
                if initial_previous_logits is None:
                    raise ValueError(
                        "Logit-aware Correction with sample_from_anchor=False "
                        "requires verifier logits for the current anchor"
                    )
                if initial_previous_logits.shape != expected_initial_shape:
                    raise ValueError(
                        "Expected initial_previous_logits shape "
                        f"{expected_initial_shape}, got "
                        f"{tuple(initial_previous_logits.shape)}"
                    )
                previous_feedback_logits = (
                    initial_previous_logits.detach()
                    .to(
                        device=dflash_hidden.device,
                        dtype=self.lm_head.weight.dtype,
                    )
                    .unsqueeze(1)
                )
                previous_feedback_mask = torch.ones(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
        elif logit_feedback_enabled and online_selector:
            if self.config.sample_from_anchor:
                if initial_previous_logits is not None:
                    raise ValueError(
                        "initial_previous_logits is only valid when "
                        "sample_from_anchor=False"
                    )
                online_previous_mask = torch.zeros(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
                assert self.candidate_selector is not None  # noqa: S101
                dummy_ids = torch.zeros(
                    dflash_hidden.shape[0],
                    1,
                    self.candidate_selector.top_k,
                    dtype=torch.long,
                    device=dflash_hidden.device,
                )
                dummy_logits = dflash_hidden.new_zeros(
                    dummy_ids.shape,
                    dtype=self.lm_head.weight.dtype,
                )
                online_previous_rank = (
                    self.correction_head.encode_previous_distribution(
                        online_previous_mask,
                        candidate_ids=dummy_ids,
                        candidate_logits=dummy_logits,
                    )
                )
            else:
                if initial_previous_logits is None:
                    raise ValueError(
                        "Logit-aware Correction with sample_from_anchor=False "
                        "requires verifier logits for the current anchor"
                    )
                online_previous_mask = torch.ones(
                    dflash_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=dflash_hidden.device,
                )
                online_previous_rank = (
                    self.correction_head.encode_previous_distribution(
                        online_previous_mask,
                        previous_logits=initial_previous_logits.detach().unsqueeze(1),
                    )
                )
        return InitialLogitFeedback(
            dense_logits=previous_feedback_logits,
            dense_mask=previous_feedback_mask,
            online_rank_features=online_previous_rank,
            online_mask=online_previous_mask,
        )

    def _rollout_correction_step(
        self,
        previous_embeddings: torch.Tensor,
        current_hidden: torch.Tensor,
        correction_hidden: torch.Tensor,
        block_positions: torch.Tensor,
        *,
        position: int,
        cache: CorrectionCache | None,
        precomputed_base_logits: torch.Tensor | None = None,
        fused_base_logits: torch.Tensor | None = None,
        **correction_kwargs: torch.Tensor,
    ) -> CorrectionStepOutput:
        """Compute one Correction residual and its final vocabulary projection.

        The caller supplies independent residual/input hidden slices to preserve
        autograd accumulation order. Selector conditioning, Markov collaboration,
        sampling and feedback-state updates remain in the rollout loop.
        """
        correction_output_mode = getattr(self.correction_head, "output_mode", "hidden")
        hidden_auxiliary_enabled = getattr(
            self.config, "correction_hidden_aux_loss", False
        )
        hidden_feedback_enabled = getattr(
            self.config, "correction_hidden_feedback", False
        )
        project_corrected_hidden = getattr(
            self.config, "correction_project_corrected_hidden", False
        )
        residual, causal_states, next_cache = self.correction_head(
            previous_embeddings,
            correction_hidden,
            block_positions,
            cache=cache,
            use_cache=True,
            **correction_kwargs,
        )
        if correction_output_mode == "logits":
            if (
                project_corrected_hidden
                or hidden_feedback_enabled
                or (hidden_auxiliary_enabled and self.training)
            ):
                delta_hidden = self.correction_head.auxiliary_hidden_residual(
                    causal_states,
                )
                corrected_current_hidden = current_hidden + delta_hidden[:, 0].to(
                    current_hidden.dtype
                )
            else:
                corrected_current_hidden = current_hidden
            projection_hidden = (
                corrected_current_hidden if project_corrected_hidden else current_hidden
            )
            if project_corrected_hidden and fused_base_logits is not None:
                hidden_delta_logits = self.correction_head.fused_lm_head_residual(
                    causal_states,
                    self.lm_head.weight,
                )
                projected_logits = fused_base_logits[:, position] + (
                    hidden_delta_logits[:, 0].to(fused_base_logits.dtype)
                )
            elif precomputed_base_logits is not None and not project_corrected_hidden:
                projected_logits = precomputed_base_logits[:, position]
            else:
                projected_logits = self.lm_head(
                    projection_hidden.to(self.lm_head.weight.dtype)
                )
            final_logits = projected_logits + residual[:, 0].to(projected_logits.dtype)
        else:
            corrected_current_hidden = current_hidden + residual[:, 0].to(
                current_hidden.dtype
            )
            if fused_base_logits is not None:
                delta_logits = self.correction_head.fused_lm_head_residual(
                    causal_states,
                    self.lm_head.weight,
                )
                final_logits = fused_base_logits[:, position] + (
                    delta_logits[:, 0].to(fused_base_logits.dtype)
                )
            else:
                final_logits = self.lm_head(
                    corrected_current_hidden.to(self.lm_head.weight.dtype)
                )
        return CorrectionStepOutput(
            logits=final_logits,
            causal_states=causal_states[:, 0],
            corrected_hidden=corrected_current_hidden,
            cache=next_cache,
        )

    def _rollout_token_embeddings(
        self,
        step_previous_ids: torch.Tensor,
        dflash_hidden: torch.Tensor,
        *,
        position: int,
        online_selector: bool,
        precomputed_base_logits: torch.Tensor | None,
        conditioning_current_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Freeze previous/current embeddings and any online Selector proposal.

        The loop chooses previous IDs before this call. Keep the previous-token
        embedding first, and preserve the Selector's argmax tie breaking and
        draft-to-verifier mapping before looking up the current token.
        """
        with torch.no_grad():
            previous_emb = self.embed_tokens(step_previous_ids).unsqueeze(1)
            if online_selector:
                assert precomputed_base_logits is not None  # noqa: S101
                online_candidate_ids, online_candidate_logits = (
                    self.dflash2_select_candidates(
                        precomputed_base_logits[:, position : position + 1],
                        dflash_hidden[:, position : position + 1],
                        step_previous_ids.unsqueeze(1),
                    )
                )
                selected_indices = online_candidate_logits.argmax(dim=-1, keepdim=True)
                selected_draft_ids = online_candidate_ids.gather(
                    -1, selected_indices
                ).squeeze(-1)
                selected_verifier_ids = self._draft_ids_to_verifier(selected_draft_ids)
                current_emb = self.embed_tokens(selected_verifier_ids)
            else:
                current_emb = (
                    self.embed_tokens(conditioning_current_ids[:, position]).unsqueeze(
                        1
                    )
                    if conditioning_current_ids is not None
                    else None
                )
        return previous_emb, current_emb

    @torch.compiler.disable
    def _rollout_correction_steps(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        initial_previous_logits: torch.Tensor | None = None,
        precomputed_base_logits: torch.Tensor | None = None,
        conditioning_current_ids: torch.Tensor | None = None,
        conditioning_previous_ids: torch.Tensor | None = None,
        conditioning_previous_rank_features: torch.Tensor | None = None,
        conditioning_previous_logits_mask: torch.Tensor | None = None,
    ) -> CorrectionRolloutOutput:
        """Run sequential Correction for rollout and offline evaluation.

        The first input is always the real anchor. By default later inputs come
        from the current Correction model.
        Static Selector conditioning supplies a complete path. Corrected feedback
        instead scores the greedy Selector online and feeds each final Correction
        token into the next slot. Main train/validation metrics use a separate
        teacher-forced path.
        """
        online_selector, has_conditioning_features = self._validate_rollout_inputs(
            dflash_hidden,
            anchor_token_ids,
            precomputed_base_logits=precomputed_base_logits,
            conditioning_current_ids=conditioning_current_ids,
            conditioning_previous_ids=conditioning_previous_ids,
            conditioning_previous_rank_features=conditioning_previous_rank_features,
            conditioning_previous_logits_mask=conditioning_previous_logits_mask,
        )
        previous_ids = anchor_token_ids.long()
        cache = None
        output_tokens: list[torch.Tensor] = []
        output_logits: list[torch.Tensor] = []
        output_states: list[torch.Tensor] = []
        output_corrected_hidden: list[torch.Tensor] = []
        start_position = 0 if self.config.sample_from_anchor else 1
        correction_output_mode = getattr(self.correction_head, "output_mode", "hidden")
        hidden_feedback_enabled = getattr(
            self.config, "correction_hidden_feedback", False
        )
        logit_feedback_enabled = correction_output_mode == "logits"
        fused_lm_head_enabled = (
            bool(getattr(self.config, "correction_lm_head_fusion", False))
            and not torch.is_grad_enabled()
        )
        fused_base_logits = None
        if fused_lm_head_enabled:
            # Project the full parallel DFlash block once. Each sequential
            # Correction position adds an inference-cached rank-to-vocabulary
            # residual instead of invoking the full LM head again.
            fused_base_logits = precomputed_base_logits
            if fused_base_logits is None:
                fused_base_logits = self.lm_head(
                    dflash_hidden.to(self.lm_head.weight.dtype)
                )
        initial_feedback = self._initial_rollout_logit_feedback(
            dflash_hidden,
            initial_previous_logits,
            logit_feedback_enabled=logit_feedback_enabled,
            has_conditioning_features=has_conditioning_features,
            online_selector=online_selector,
        )
        feedback = RolloutFeedbackState.from_initial(
            initial_feedback,
            dflash_hidden,
            hidden_feedback_enabled=hidden_feedback_enabled,
        )
        del initial_feedback  # Do not retain superseded dense logits through rollout.

        for position in range(self.block_size):
            current_hidden = dflash_hidden[:, position]
            if position < start_position:
                corrected_current_hidden = current_hidden
                available_base_logits = (
                    fused_base_logits
                    if fused_base_logits is not None
                    else precomputed_base_logits
                )
                if available_base_logits is not None:
                    final_logits = available_base_logits[:, position]
                else:
                    final_logits = self.lm_head(
                        current_hidden.to(self.lm_head.weight.dtype)
                    )
                causal_states = current_hidden.new_zeros(
                    current_hidden.shape[0], self.config.correction_hidden_size
                )
            else:
                step_previous_ids = (
                    conditioning_previous_ids[:, position]
                    if conditioning_previous_ids is not None
                    else previous_ids
                )
                previous_emb, current_emb = self._rollout_token_embeddings(
                    step_previous_ids,
                    dflash_hidden,
                    position=position,
                    online_selector=online_selector,
                    precomputed_base_logits=precomputed_base_logits,
                    conditioning_current_ids=conditioning_current_ids,
                )
                current_token_kwargs = (
                    {}
                    if current_emb is None
                    else {"current_token_embeddings": current_emb}
                )
                block_positions = torch.full(
                    (dflash_hidden.shape[0], 1),
                    position,
                    dtype=torch.long,
                    device=dflash_hidden.device,
                )
                feedback_kwargs = feedback.correction_kwargs(
                    position,
                    hidden_feedback_enabled=hidden_feedback_enabled,
                    logit_feedback_enabled=logit_feedback_enabled,
                    has_conditioning_features=has_conditioning_features,
                    online_selector=online_selector,
                    conditioning_previous_rank_features=conditioning_previous_rank_features,
                    conditioning_previous_logits_mask=conditioning_previous_logits_mask,
                )
                step = self._rollout_correction_step(
                    previous_emb,
                    current_hidden,
                    dflash_hidden[:, position : position + 1],
                    block_positions,
                    position=position,
                    cache=cache,
                    precomputed_base_logits=precomputed_base_logits,
                    fused_base_logits=fused_base_logits,
                    **current_token_kwargs,
                    **feedback_kwargs,
                )
                final_logits = step.logits
                causal_states = step.causal_states
                corrected_current_hidden = step.corrected_hidden
                cache = step.cache
                del step  # Match the previous tuple-unpacking tensor lifetimes.
                if getattr(self, "markov_head", None) is not None:
                    final_logits, _, _ = self._apply_collaborative_markov(
                        final_logits.unsqueeze(1),
                        causal_states.unsqueeze(1),
                        step_previous_ids.unsqueeze(1),
                        dflash_hidden[:, position : position + 1],
                    )
                    final_logits = final_logits[:, 0]

            with torch.no_grad():
                if temperature > 0:
                    probabilities = torch.softmax(
                        final_logits.float() / temperature, dim=-1
                    )
                    sampled_indices = torch.multinomial(
                        probabilities, num_samples=1
                    ).squeeze(-1)
                else:
                    sampled_indices = torch.argmax(final_logits, dim=-1)
                draft_ids = sampled_indices
            output_tokens.append(draft_ids)
            output_logits.append(final_logits)
            output_states.append(causal_states)
            output_corrected_hidden.append(corrected_current_hidden)

            feedback.advance(
                final_logits,
                corrected_current_hidden,
                position=position,
                start_position=start_position,
                logit_feedback_enabled=logit_feedback_enabled,
                has_conditioning_features=has_conditioning_features,
                online_selector=online_selector,
                hidden_feedback_enabled=hidden_feedback_enabled,
                correction_head=self.correction_head,
            )
            if position < start_position:
                continue
            if conditioning_previous_ids is None:
                previous_ids = self._draft_ids_to_verifier(draft_ids)

        return CorrectionRolloutOutput(
            token_ids=torch.stack(output_tokens, dim=1),
            logits=torch.stack(output_logits, dim=1),
            causal_states=torch.stack(output_states, dim=1),
            corrected_hidden=torch.stack(output_corrected_hidden, dim=1),
        )

    def _validation_correction_outputs(
        self,
        hidden: torch.Tensor,
        hidden_blocks: torch.Tensor,
        *,
        base_logits: torch.Tensor | None,
        base_logits_blocks: torch.Tensor | None,
        targets: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        correction_output_mode: str | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return optional validation base diagnostics and flattened rollout logits."""
        if self.training:
            return base_logits, None

        # This extra projection is for diagnostics only, not rollout conditioning.
        if self.config.correction_base_diagnostics and base_logits is None:
            with torch.no_grad():
                base_logits = self.lm_head(
                    hidden.detach().to(self.lm_head.weight.dtype)
                )

        rollout_logits = None
        if self.config.correction_rollout_metrics:
            num_blocks, block = hidden_blocks.shape[:2]
            rollout_initial_logits = None
            if (
                not self.config.sample_from_anchor
                and correction_output_mode == "logits"
            ):
                rollout_initial_logits = targets.view(num_blocks, block, -1)[:, 0]
            _, rollout_blocks = self.rollout_correction(
                hidden_blocks.detach(),
                anchor_token_ids=anchor_token_ids,
                initial_previous_logits=rollout_initial_logits,
                base_logits=base_logits_blocks,
            )
            rollout_logits = rollout_blocks.reshape(1, num_blocks * block, -1)
        return base_logits, rollout_logits

    def _add_auxiliary_losses(
        self,
        loss: torch.Tensor,
        metrics: dict[str, torch.Tensor],
        *,
        selector_loss: torch.Tensor | None,
        corrected_hidden: torch.Tensor | None,
        verifier_last_hidden_states: torch.Tensor,
        anchored_block_indices: torch.Tensor,
        aligned_loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add Selector then hidden-alignment losses, updating metrics in place."""
        if selector_loss is not None:
            loss = loss + self.config.dflash2_selector_loss_weight * selector_loss
            metrics["loss_sum"] = loss.detach().clone()
            metrics["dflash2_selector_loss_sum"] = selector_loss.detach().clone()
            metrics["dflash2_selector_loss_total"] = torch.ones(
                (), device=loss.device, dtype=torch.float32
            )
        if self.config.correction_hidden_aux_loss:
            if corrected_hidden is None:
                raise RuntimeError(
                    "Hidden auxiliary loss requires corrected DFlash hidden states"
                )
            with torch.no_grad():
                verifier_hidden_targets = self.verifier_norm(
                    verifier_last_hidden_states.to(self.verifier_norm.weight.dtype)
                )
                if not self.config.sample_from_anchor:
                    verifier_hidden_targets = torch.roll(
                        verifier_hidden_targets, 1, dims=1
                    )
                verifier_hidden_targets = verifier_hidden_targets[
                    :, anchored_block_indices
                ].view_as(corrected_hidden)
            hidden_aux_loss = self._hidden_alignment_loss(
                corrected_hidden,
                verifier_hidden_targets,
                aligned_loss_mask.view(*corrected_hidden.shape[:2]),
            )
            loss = loss + (self.config.correction_hidden_aux_weight * hidden_aux_loss)
            metrics["loss_sum"] = loss.detach().clone()
            metrics["correction_hidden_aux_loss_sum"] = hidden_aux_loss.detach().clone()
            metrics["correction_hidden_aux_loss_total"] = torch.ones(
                (),
                device=loss.device,
                dtype=torch.float32,
            )
        return loss, metrics

    def _run_teacher_forced_correction(
        self,
        blocks: TrainingBlocks,
        targets: torch.Tensor,
        conditioning: SelectorConditioning,
        *,
        correction_output_mode: str | None,
    ) -> TeacherForcedCorrectionOutput:
        """Dispatch Correction before collaboration, validation outputs and losses.

        The caller has already enabled Correction and prepared Selector inputs.
        Frozen previous-token lookup precedes recurrent logit-feature preparation;
        the parallel hidden-output path retains its single final LM projection.
        """
        num_blocks, block = blocks.hidden.shape[:2]
        mask_tokens_size = num_blocks * block
        if self.config.correction_hidden_feedback:
            with torch.no_grad():
                prev_gt_emb = self.embed_tokens(conditioning.previous_token_ids)
            previous_target_logits = None
            previous_target_mask = None
            previous_rank_features = None
            if correction_output_mode == "logits":
                if conditioning.previous_logits_mask is not None:
                    previous_rank_features = conditioning.previous_rank_features
                    previous_target_mask = conditioning.previous_logits_mask
                else:
                    target_blocks = targets.view(num_blocks, block, -1)
                    previous_target_logits = torch.cat(
                        [
                            torch.zeros_like(target_blocks[:, :1]),
                            target_blocks[:, :-1],
                        ],
                        dim=1,
                    )
                    previous_target_mask = blocks.positions > 0
            logits_blocks, correction_states, corrected_hidden = (
                self._teacher_forced_hidden_feedback_correction(
                    blocks.hidden,
                    prev_gt_emb,
                    blocks.positions,
                    blocks.base_logits,
                    previous_target_logits,
                    previous_target_mask,
                    previous_rank_features=previous_rank_features,
                    current_token_embeddings=conditioning.current_token_embeddings,
                )
            )
            logits = logits_blocks.reshape(1, mask_tokens_size, -1)
        else:
            logits, correction_states, corrected_hidden = (
                self._teacher_forced_parallel_correction(
                    blocks.hidden,
                    conditioning.previous_token_ids,
                    blocks.positions,
                    targets=targets,
                    base_logits_blocks=blocks.base_logits,
                    selector_current_embeddings=conditioning.current_token_embeddings,
                    selector_previous_rank_features=conditioning.previous_rank_features,
                    selector_previous_logits_mask=conditioning.previous_logits_mask,
                )
            )
        if (
            correction_output_mode == "hidden"
            and not self.config.correction_hidden_feedback
        ):
            logits = self.lm_head(
                corrected_hidden.reshape(1, mask_tokens_size, -1).to(
                    self.lm_head.weight.dtype
                )
            )
        return TeacherForcedCorrectionOutput(
            logits=logits,
            causal_states=correction_states,
            corrected_hidden=corrected_hidden,
        )

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 7.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        confidence_length_alpha: float = 0.0,
        confidence_loss_weighting: str = "match-draft",
        first_error_focal_alpha: float = 0.0,
        adaptive_loss: str = "none",
        ssal_decay_weight: torch.Tensor | float = 0.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        correction_output_mode = (
            getattr(self.correction_head, "output_mode", "hidden")
            if self.correction_head is not None
            else None
        )
        (
            hidden,
            logits,
            targets,
            aligned_loss_mask,
            anchored_block_indices,
            target_log_normalizer,
            target_argmax_ids,
        ) = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            project_logits=(
                self.correction_head is None
                or self.candidate_selector is not None
                or (
                    correction_output_mode == "logits"
                    and not self.config.correction_project_corrected_hidden
                )
            ),
            **kwargs,
        )

        # DSpark: add the active sequential correction and predict confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        base_logits = logits
        blocks = prepare_training_blocks(
            input_ids,
            anchored_block_indices,
            hidden,
            base_logits,
            num_blocks=num_blocks,
            block_size=block,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        selector_conditioning = self._prepare_selector_conditioning(
            base_logits,
            targets,
            blocks.hidden,
            block_tokens=blocks.token_ids,
            aligned_loss_mask=aligned_loss_mask,
            prev_token_ids=blocks.previous_token_ids,
            block_positions=blocks.positions,
            correction_output_mode=correction_output_mode,
        )
        selector_loss = selector_conditioning.selector_loss
        confidence_logits = None
        prev_emb = None
        correction_states = None
        rollout_logits = None
        collaboration_base_logits = None
        collaboration_gate = None
        corrected_hidden = None
        if self.correction_head is not None:
            corrected = self._run_teacher_forced_correction(
                blocks,
                targets,
                selector_conditioning,
                correction_output_mode=correction_output_mode,
            )
            logits = corrected.logits
            correction_states = corrected.causal_states
            corrected_hidden = corrected.corrected_hidden
            del corrected  # Do not retain pre-collaboration logits in a wrapper.
            if self.markov_head is not None:
                collaboration_base_logits = logits
                collaborative_blocks, collaboration_gate, prev_emb = (
                    self._apply_collaborative_markov(
                        logits.view(num_blocks, block, -1),
                        correction_states,
                        selector_conditioning.previous_token_ids,
                        blocks.hidden,
                    )
                )
                logits = collaborative_blocks.reshape(1, mask_tokens_size, -1)

            base_logits, rollout_logits = self._validation_correction_outputs(
                hidden,
                blocks.hidden,
                base_logits=base_logits,
                base_logits_blocks=blocks.base_logits,
                targets=targets,
                anchor_token_ids=blocks.token_ids[:, 0],
                correction_output_mode=correction_output_mode,
            )
        elif self.markov_head is not None:
            if logits is None:
                raise RuntimeError("Markov correction requires base logits")
            prev_emb = self.markov_head.prev_embeddings(blocks.previous_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=blocks.previous_token_ids,
                hidden_states=blocks.hidden,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if logits is None:
            raise RuntimeError("DSpark forward did not produce draft logits")

        proposal_candidate_ids = None
        proposal_candidate_logits = None
        if self.candidate_selector is not None and self.correction_head is None:
            candidate_ids, candidate_logits, selector_loss, selected_ids, _ = (
                self._dflash2_block_outputs(
                    logits,
                    targets,
                    blocks.hidden,
                    blocks.token_ids[:, 0],
                    aligned_loss_mask,
                    teacher_previous_token_ids=blocks.previous_token_ids,
                )
            )
            candidate_logits = self._dflash2_proposal_logits(
                candidate_ids,
                candidate_logits,
                selected_ids,
            )
            proposal_candidate_ids = candidate_ids.view(1, mask_tokens_size, -1)
            proposal_candidate_logits = candidate_logits.view_as(proposal_candidate_ids)

        if self.confidence_head is not None:
            sequential_states = None
            if self.config.confidence_head_with_markov:
                sequential_states = (
                    correction_states if correction_states is not None else prev_emb
                )
            conf_features = self._confidence_features(
                blocks.hidden,
                sequential_states,
                detach=self.config.confidence_detach_features,
            )
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            confidence_length_alpha=confidence_length_alpha,
            confidence_loss_weighting=confidence_loss_weighting,  # type: ignore[arg-type]
            first_error_focal_alpha=first_error_focal_alpha,
            adaptive_loss=adaptive_loss,  # type: ignore[arg-type]
            ssal_decay_weight=ssal_decay_weight,
            base_logits=(
                base_logits
                if self.markov_head is not None
                or (self.correction_head is not None and base_logits is not None)
                else None
            ),
            rollout_logits=rollout_logits,
            collaboration_base_logits=collaboration_base_logits,
            collaboration_gate=collaboration_gate,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            proposal_candidate_ids=proposal_candidate_ids,
            proposal_candidate_logits=proposal_candidate_logits,
            target_log_normalizer=target_log_normalizer,
            target_argmax_ids=target_argmax_ids,
        )
        loss, metrics = self._add_auxiliary_losses(
            loss,
            metrics,
            selector_loss=selector_loss,
            corrected_hidden=corrected_hidden,
            verifier_last_hidden_states=verifier_last_hidden_states,
            anchored_block_indices=anchored_block_indices,
            aligned_loss_mask=aligned_loss_mask,
        )
        metrics = select_logged_metrics(
            metrics,
            include_diagnostics=(
                not self.training and self.config.correction_base_diagnostics
            ),
        )
        return None, loss, metrics

    @torch.compiler.disable
    @torch.no_grad()
    def rollout_correction(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        initial_previous_logits: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Correction using a static selector path or online feedback."""
        conditioning_previous_ids = None
        conditioning_current_ids = None
        conditioning_previous_rank_features = None
        conditioning_previous_logits_mask = None
        if self.candidate_selector is not None:
            if base_logits is None:
                base_logits = self.lm_head(dflash_hidden.to(self.lm_head.weight.dtype))
            if (
                getattr(self.config, "selector_correction_feedback", "static")
                == "static"
            ):
                candidate_ids, selector_logits, selected_ids = self.dflash2_select_path(
                    base_logits,
                    dflash_hidden,
                    anchor_token_ids,
                )
                selector_inputs = self._selector_correction_inputs(
                    candidate_ids,
                    selector_logits,
                    selected_ids,
                    anchor_token_ids,
                    initial_previous_logits=initial_previous_logits,
                )
                conditioning_current_ids = selector_inputs.current_token_ids
                conditioning_previous_ids = selector_inputs.previous_token_ids
                conditioning_previous_rank_features = (
                    selector_inputs.previous_rank_features
                )
                conditioning_previous_logits_mask = selector_inputs.previous_logits_mask
        rollout = self._rollout_correction_steps(
            dflash_hidden,
            anchor_token_ids,
            temperature=temperature,
            initial_previous_logits=initial_previous_logits,
            precomputed_base_logits=base_logits,
            conditioning_current_ids=conditioning_current_ids,
            conditioning_previous_ids=conditioning_previous_ids,
            conditioning_previous_rank_features=(conditioning_previous_rank_features),
            conditioning_previous_logits_mask=conditioning_previous_logits_mask,
        )
        return rollout.token_ids, rollout.logits
