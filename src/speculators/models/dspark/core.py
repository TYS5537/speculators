from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics, select_logged_metrics
from speculators.models.dspark.model_definitions import (
    CausalCorrectionHead,
    ConfidenceHead,
    MarkovHead,
)
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

_DSPARK_PAPER_LOSS_FN = '{"ce": 0.1, "tv": 0.9}'
_DEFAULT_LOSS_CONFIG: LossConfig = resolve_loss_config(_DSPARK_PAPER_LOSS_FN)

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a sequential correction and confidence head.

    The legacy Markov path refines base logits. The causal Correction path can
    either refine DFlash hidden states before the sole LM-head projection or
    consume previous logits and refine base logits with a low-rank vocabulary
    bias. An opt-in collaboration path gates a further Markov bias from Correction
    state. Optional hidden alignment and corrected-hidden feedback provide
    representation-level supervision and recurrence. The confidence head predicts
    each position's acceptance probability.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size
        if (
            config.dflash2_candidate_selector
            and config.dflash2_selector_search_mode == "global"
            and config.markov_rank > 0
            and not config.enable_correction_head
        ):
            raise ValueError(
                "Global DFlash2 path search must run before Correction and is not "
                "compatible with a standalone predecessor-dependent Markov head"
            )
        if config.selector_correction_feedback == "corrected":
            if (
                not config.dflash2_candidate_selector
                or not config.enable_correction_head
            ):
                raise ValueError(
                    "selector_correction_feedback='corrected' requires Selector "
                    "and Correction"
                )
            if config.dflash2_selector_search_mode != "greedy":
                raise ValueError(
                    "selector_correction_feedback='corrected' requires greedy "
                    "Selector search"
                )
        if (
            config.correction_output_mode != "hidden"
            and not config.enable_correction_head
        ):
            raise ValueError("correction_output_mode='logits' requires Correction")
        if config.correction_lm_head_fusion:
            if not config.enable_correction_head:
                raise ValueError("correction_lm_head_fusion=True requires Correction")
            if (
                config.correction_output_mode == "logits"
                and not config.correction_project_corrected_hidden
            ):
                raise ValueError(
                    "Logit Correction LM-head fusion requires "
                    "correction_project_corrected_hidden=True"
                )
        if (
            config.correction_hidden_aux_loss
            or config.correction_hidden_feedback
            or config.correction_project_corrected_hidden
        ) and not config.enable_correction_head:
            raise ValueError(
                "Correction auxiliary/feedback features require Correction"
            )
        if (
            config.correction_project_corrected_hidden
            and config.correction_output_mode != "logits"
        ):
            raise ValueError("correction_project_corrected_hidden requires logits mode")

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
                if config.markov_rank <= 0:
                    raise ValueError(
                        "correction_with_markov=True requires markov_rank > 0"
                    )
                if config.markov_head_type == "rnn":
                    raise ValueError(
                        "Correction-Markov collaboration supports only vanilla "
                        "or gated Markov heads"
                    )
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
            if (
                config.confidence_head_with_markov
                and self.markov_head is None
                and self.correction_head is None
            ):
                raise ValueError(
                    "confidence_head_with_markov=True requires an enabled Markov "
                    "or correction head."
                )
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
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        kwargs.setdefault("block_size", 7)
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_correction_head=kwargs.get("enable_correction_head", False),
            correction_output_mode=kwargs.get("correction_output_mode", "hidden"),
            correction_hidden_size=kwargs.get("correction_hidden_size", 512),
            correction_rank=kwargs.get("correction_rank", 256),
            correction_lm_head_fusion=kwargs.get("correction_lm_head_fusion", False),
            correction_num_layers=kwargs.get("correction_num_layers", 1),
            correction_num_heads=kwargs.get("correction_num_heads", 8),
            correction_gate_bias=kwargs.get("correction_gate_bias", 0.0),
            correction_hidden_aux_loss=kwargs.get("correction_hidden_aux_loss", False),
            correction_hidden_aux_weight=kwargs.get(
                "correction_hidden_aux_weight", 0.1
            ),
            correction_hidden_feedback=kwargs.get("correction_hidden_feedback", False),
            selector_correction_feedback=kwargs.get(
                "selector_correction_feedback", "static"
            ),
            correction_project_corrected_hidden=kwargs.get(
                "correction_project_corrected_hidden", False
            ),
            correction_with_markov=kwargs.get("correction_with_markov", False),
            correction_markov_gate_bias=kwargs.get("correction_markov_gate_bias", -2.0),
            correction_rollout_metrics=kwargs.get("correction_rollout_metrics", False),
            correction_base_diagnostics=kwargs.get(
                "correction_base_diagnostics", False
            ),
            enable_confidence_head=(
                True
                if enable_confidence_head_arg is None
                else enable_confidence_head_arg
            ),
            confidence_head_with_markov=(
                True
                if confidence_head_with_markov_arg is None
                else confidence_head_with_markov_arg
            ),
            confidence_detach_features=kwargs.get("confidence_detach_features", False),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config = resolve_loss_config(kwargs.get("loss_fn", _DSPARK_PAPER_LOSS_FN))
        gamma = kwargs.get("dflash_decay_gamma", float(kwargs.get("block_size", 7)))
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        confidence_length_alpha = kwargs.get("confidence_length_alpha", 0.0)
        confidence_loss_weighting = kwargs.get(
            "confidence_loss_weighting", "match-draft"
        )
        first_error_focal_alpha = kwargs.get("first_error_focal_alpha", 0.0)
        adaptive_loss = kwargs.get("adaptive_loss", "none")
        ssal_curriculum = kwargs.get("ssal_curriculum", False)
        ssal_curriculum_start = kwargs.get("ssal_curriculum_start", 0.1)
        ssal_curriculum_end = kwargs.get("ssal_curriculum_end", 0.6)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "confidence_head_alpha": confidence_head_alpha,
            "confidence_length_alpha": confidence_length_alpha,
            "confidence_loss_weighting": confidence_loss_weighting,
            "first_error_focal_alpha": first_error_focal_alpha,
            "adaptive_loss": adaptive_loss,
            "ssal_decay_weight": 0.0,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        train_kw = dict(shared)
        if ssal_curriculum:
            train_kw["ssal_curriculum"] = True
            train_kw["ssal_curriculum_start"] = ssal_curriculum_start
            train_kw["ssal_curriculum_end"] = ssal_curriculum_end
        return train_kw, dict(shared)

    @staticmethod
    def _confidence_features(
        hidden_states: torch.Tensor,
        sequential_states: torch.Tensor | None,
        *,
        detach: bool,
    ) -> torch.Tensor:
        """Build consistently coupled or detached confidence-head features."""
        if detach:
            hidden_states = hidden_states.detach()
            if sequential_states is not None:
                sequential_states = sequential_states.detach()
        if sequential_states is None:
            return hidden_states
        return torch.cat(
            [hidden_states, sequential_states.to(hidden_states.dtype)], dim=-1
        )

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
        """Run teacher-forced Correction with corrected-hidden recurrence."""
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
                if current_token_embeddings is not None:
                    head_kwargs["current_token_embeddings"] = current_token_embeddings[
                        :, position : position + 1
                    ]
                needs_previous_logits = self.correction_head.output_mode == "logits"
                if needs_previous_logits:
                    if previous_target_logits_mask is None:
                        raise RuntimeError(
                            "Logit-aware Correction requires previous feature masks"
                        )
                    head_kwargs["previous_logits_mask"] = previous_target_logits_mask[
                        :, position : position + 1
                    ]
                    if previous_rank_features is not None:
                        head_kwargs["previous_rank_features"] = (
                            previous_rank_features[:, position : position + 1]
                        )
                    else:
                        if previous_target_logits is None:
                            raise RuntimeError(
                                "Logit-aware Correction requires previous logits"
                            )
                        head_kwargs["previous_logits"] = previous_target_logits[
                            :, position : position + 1
                        ]
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
        return logits, correction_states, corrected_hidden

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

    def _selector_correction_inputs(
        self,
        candidate_ids: torch.Tensor,
        selector_logits: torch.Tensor,
        selected_draft_ids: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        initial_previous_logits: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Build token IDs and compact previous-distribution features."""
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
        current_ids = self._draft_ids_to_verifier(selected_draft_ids)
        previous_ids = anchor_token_ids[:, None].expand(-1, block_size).clone().long()
        start_position = 0 if self.config.sample_from_anchor else 1
        if start_position + 1 < block_size:
            previous_ids[:, start_position + 1 :] = self._draft_ids_to_verifier(
                selected_draft_ids[:, start_position:-1]
            )

        needs_logit_features = (
            getattr(self.correction_head, "output_mode", "hidden") == "logits"
        )
        if not needs_logit_features:
            return current_ids, previous_ids, None, None

        previous_candidate_ids = torch.zeros_like(candidate_ids)
        previous_candidate_logits = torch.zeros_like(selector_logits)
        sparse_mask = torch.zeros(
            num_blocks,
            block_size,
            dtype=torch.bool,
            device=candidate_ids.device,
        )
        if self.config.sample_from_anchor:
            if initial_previous_logits is not None:
                raise ValueError(
                    "Selector initial logits are only valid when "
                    "sample_from_anchor=False"
                )
            if block_size > 1:
                previous_candidate_ids[:, 1:] = candidate_ids[:, :-1]
                previous_candidate_logits[:, 1:] = selector_logits[:, :-1]
                sparse_mask[:, 1:] = True
        else:
            if start_position + 1 < block_size:
                previous_candidate_ids[:, start_position + 1 :] = candidate_ids[
                    :, start_position:-1
                ]
                previous_candidate_logits[:, start_position + 1 :] = selector_logits[
                    :, start_position:-1
                ]
                sparse_mask[:, start_position + 1 :] = True

        previous_rank = self.correction_head.encode_previous_distribution(
            sparse_mask,
            candidate_ids=previous_candidate_ids,
            candidate_logits=previous_candidate_logits,
        )
        previous_logits_mask = sparse_mask
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
            previous_logits_mask = sparse_mask.clone()
            previous_logits_mask[:, start_position] = True
        return (
            current_ids,
            previous_ids,
            previous_rank,
            previous_logits_mask,
        )

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run sequential Correction for rollout and offline evaluation.

        The first input is always the real anchor. By default later inputs come
        from the current Correction model.
        Static Selector conditioning supplies a complete path. Corrected feedback
        instead scores the greedy Selector online and feeds each final Correction
        token into the next slot. Main train/validation metrics use a separate
        teacher-forced path.
        """
        if self.correction_head is None:
            raise RuntimeError(
                "_rollout_correction_steps requires enable_correction_head=True"
            )
        if dflash_hidden.ndim != 3:
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
        if online_selector and conditioning_current_ids is not None:
            raise ValueError("Corrected Selector feedback cannot use a static path")
        if conditioning_current_ids is not None and (
            conditioning_current_ids.shape != expected_block_shape
        ):
            raise ValueError(
                "Expected conditioning_current_ids shape "
                f"{expected_block_shape}, got {tuple(conditioning_current_ids.shape)}"
            )
        if (conditioning_current_ids is None) != (conditioning_previous_ids is None):
            raise ValueError(
                "Selector current and previous token IDs must be provided together"
            )
        if conditioning_previous_ids is not None and (
            conditioning_previous_ids.shape != expected_block_shape
        ):
            raise ValueError(
                "Expected conditioning_previous_ids shape "
                f"{expected_block_shape}, got {tuple(conditioning_previous_ids.shape)}"
            )
        has_conditioning_features = conditioning_previous_rank_features is not None
        if has_conditioning_features != (conditioning_previous_logits_mask is not None):
            raise ValueError(
                "Selector compact logit features and mask must be provided together"
            )
        if has_conditioning_features:
            assert conditioning_previous_logits_mask is not None  # noqa: S101
            if conditioning_previous_logits_mask.shape != expected_block_shape:
                raise ValueError(
                    "Selector previous-logit mask must align with block positions"
                )
            if conditioning_previous_rank_features is not None and (
                conditioning_previous_rank_features.shape[:-1] != expected_block_shape
            ):
                raise ValueError("Selector previous-rank features must align")
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

        previous_ids = anchor_token_ids.long()
        cache = None
        output_tokens: list[torch.Tensor] = []
        output_logits: list[torch.Tensor] = []
        output_states: list[torch.Tensor] = []
        output_corrected_hidden: list[torch.Tensor] = []
        start_position = 0 if self.config.sample_from_anchor else 1
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
        logit_feedback_enabled = correction_output_mode == "logits"
        fused_lm_head_enabled = bool(
            getattr(self.config, "correction_lm_head_fusion", False)
        ) and not torch.is_grad_enabled()
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
        previous_corrected_hidden = None
        previous_corrected_hidden_mask = None
        if hidden_feedback_enabled:
            previous_corrected_hidden = dflash_hidden.new_zeros(
                dflash_hidden.shape[0],
                1,
                dflash_hidden.shape[-1],
            )
            previous_corrected_hidden_mask = torch.zeros(
                dflash_hidden.shape[0],
                1,
                dtype=torch.bool,
                device=dflash_hidden.device,
            )

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
                online_candidate_ids = None
                online_candidate_logits = None
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
                        selected_indices = online_candidate_logits.argmax(
                            dim=-1, keepdim=True
                        )
                        selected_draft_ids = online_candidate_ids.gather(
                            -1, selected_indices
                        ).squeeze(-1)
                        selected_verifier_ids = self._draft_ids_to_verifier(
                            selected_draft_ids
                        )
                        current_emb = self.embed_tokens(selected_verifier_ids)
                    else:
                        current_emb = (
                            self.embed_tokens(
                                conditioning_current_ids[:, position]
                            ).unsqueeze(1)
                            if conditioning_current_ids is not None
                            else None
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
                hidden_feedback_kwargs = {}
                if hidden_feedback_enabled:
                    assert previous_corrected_hidden is not None  # noqa: S101
                    assert previous_corrected_hidden_mask is not None  # noqa: S101
                    hidden_feedback_kwargs = {
                        "previous_corrected_hidden": previous_corrected_hidden,
                        "previous_corrected_hidden_mask": (
                            previous_corrected_hidden_mask
                        ),
                    }
                logit_feedback_kwargs = {}
                if logit_feedback_enabled:
                    if has_conditioning_features:
                        assert (  # noqa: S101
                            conditioning_previous_logits_mask is not None
                        )
                        logit_feedback_kwargs["previous_logits_mask"] = (
                            conditioning_previous_logits_mask[
                                :, position : position + 1
                            ]
                        )
                        if conditioning_previous_rank_features is not None:
                            logit_feedback_kwargs["previous_rank_features"] = (
                                conditioning_previous_rank_features[
                                    :, position : position + 1
                                ]
                            )
                    elif online_selector:
                        assert online_previous_mask is not None  # noqa: S101
                        logit_feedback_kwargs = {
                            "previous_logits_mask": online_previous_mask,
                        }
                        if online_previous_rank is not None:
                            logit_feedback_kwargs["previous_rank_features"] = (
                                online_previous_rank
                            )
                    else:
                        assert previous_feedback_logits is not None  # noqa: S101
                        assert previous_feedback_mask is not None  # noqa: S101
                        logit_feedback_kwargs = {
                            "previous_logits": previous_feedback_logits,
                            "previous_logits_mask": previous_feedback_mask,
                        }
                if correction_output_mode == "logits":
                    delta_logits, causal_states, cache = self.correction_head(
                        previous_emb,
                        dflash_hidden[:, position : position + 1],
                        block_positions,
                        cache=cache,
                        use_cache=True,
                        **current_token_kwargs,
                        **hidden_feedback_kwargs,
                        **logit_feedback_kwargs,
                    )
                    if (
                        project_corrected_hidden
                        or hidden_feedback_enabled
                        or (hidden_auxiliary_enabled and self.training)
                    ):
                        delta_hidden = self.correction_head.auxiliary_hidden_residual(
                            causal_states,
                        )
                        corrected_current_hidden = current_hidden + delta_hidden[
                            :, 0
                        ].to(current_hidden.dtype)
                    else:
                        corrected_current_hidden = current_hidden
                    projection_hidden = (
                        corrected_current_hidden
                        if project_corrected_hidden
                        else current_hidden
                    )
                    if project_corrected_hidden and fused_base_logits is not None:
                        hidden_delta_logits = (
                            self.correction_head.fused_lm_head_residual(
                                causal_states,
                                self.lm_head.weight,
                            )
                        )
                        projected_logits = fused_base_logits[:, position] + (
                            hidden_delta_logits[:, 0].to(fused_base_logits.dtype)
                        )
                    elif (
                        precomputed_base_logits is not None
                        and not project_corrected_hidden
                    ):
                        projected_logits = precomputed_base_logits[:, position]
                    else:
                        projected_logits = self.lm_head(
                            projection_hidden.to(self.lm_head.weight.dtype)
                        )
                    final_logits = projected_logits + delta_logits[:, 0].to(
                        projected_logits.dtype
                    )
                else:
                    delta_hidden, causal_states, cache = self.correction_head(
                        previous_emb,
                        dflash_hidden[:, position : position + 1],
                        block_positions,
                        cache=cache,
                        use_cache=True,
                        **current_token_kwargs,
                        **hidden_feedback_kwargs,
                        **logit_feedback_kwargs,
                    )
                    corrected_current_hidden = current_hidden + delta_hidden[:, 0].to(
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
                causal_states = causal_states[:, 0]
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

            if (
                logit_feedback_enabled
                and not has_conditioning_features
                and not online_selector
                and position >= start_position
            ):
                previous_feedback_logits = final_logits.detach().unsqueeze(1)
                previous_feedback_mask = torch.ones(
                    final_logits.shape[0],
                    1,
                    dtype=torch.bool,
                    device=final_logits.device,
                )
            if (
                online_selector
                and logit_feedback_enabled
                and position >= start_position
            ):
                online_previous_mask = torch.ones(
                    final_logits.shape[0],
                    1,
                    dtype=torch.bool,
                    device=final_logits.device,
                )
                online_previous_rank = (
                    self.correction_head.encode_previous_distribution(
                        online_previous_mask,
                        previous_logits=final_logits.detach().unsqueeze(1),
                    )
                )
            if hidden_feedback_enabled:
                previous_corrected_hidden = corrected_current_hidden.unsqueeze(1)
                previous_corrected_hidden_mask = torch.ones(
                    corrected_current_hidden.shape[0],
                    1,
                    dtype=torch.bool,
                    device=corrected_current_hidden.device,
                )
            if position < start_position:
                continue
            if conditioning_previous_ids is None:
                previous_ids = self._draft_ids_to_verifier(draft_ids)

        return (
            torch.stack(output_tokens, dim=1),
            torch.stack(output_logits, dim=1),
            torch.stack(output_states, dim=1),
            torch.stack(output_corrected_hidden, dim=1),
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
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
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
        )

        # DSpark: add the active sequential correction and predict confidence.
        num_blocks = max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        base_logits = logits
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        if self.config.sample_from_anchor:
            # With sample_from_anchor=True (DSpark default), slot k predicts
            # token p+k+1 and the inference Markov chain conditions slot k's
            # bias on the token at the previous position p+k.
            prev_token_ids = block_tokens
        else:
            # With sample_from_anchor=False (Dflash default), slot k predicts
            # token p+k, so the previous token within the block is
            # block_tokens[:, k-1] (shifted).
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)
        block_positions = torch.arange(block, device=hidden.device).expand(
            num_blocks, -1
        )
        base_logits_blocks = (
            None if base_logits is None else base_logits.view(num_blocks, block, -1)
        )
        selector_loss = None
        selector_candidate_ids = None
        selector_candidate_logits = None
        selector_teacher_logits = None
        selector_selected_ids = None
        selector_current_ids = None
        selector_previous_ids = None
        selector_previous_rank_features = None
        selector_previous_logits_mask = None
        if self.candidate_selector is not None and self.correction_head is not None:
            if base_logits is None or base_logits_blocks is None:
                raise RuntimeError(
                    "Selector-conditioned Correction requires pure DFlash base logits"
                )
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
                        device=hidden.device,
                    )
                    teacher_rank = self.correction_head.encode_previous_distribution(
                        teacher_source_mask,
                        previous_logits=targets.view(num_blocks, block, -1)[:, :-1],
                    )

                    selector_previous_rank_features = (
                        self._prepend_zero_compact_feature(teacher_rank)
                    )
                    selector_previous_logits_mask = block_positions > 0
            else:
                (
                    selector_current_ids,
                    selector_previous_ids,
                    selector_previous_rank_features,
                    selector_previous_logits_mask,
                ) = self._selector_correction_inputs(
                    selector_candidate_ids,
                    selector_candidate_logits,
                    selector_selected_ids,
                    block_tokens[:, 0],
                    initial_previous_logits=selector_initial_logits,
                )
        selector_current_embeddings = None
        if selector_current_ids is not None:
            with torch.no_grad():
                selector_current_embeddings = self.embed_tokens(selector_current_ids)
        selector_current_kwargs = (
            {}
            if selector_current_embeddings is None
            else {"current_token_embeddings": selector_current_embeddings}
        )
        selector_current_tail_kwargs = (
            {}
            if selector_current_embeddings is None
            else {"current_token_embeddings": selector_current_embeddings[:, 1:]}
        )
        correction_previous_ids = (
            selector_previous_ids
            if selector_previous_ids is not None
            else prev_token_ids
        )
        confidence_logits = None
        prev_emb = None
        correction_states = None
        rollout_logits = None
        collaboration_base_logits = None
        collaboration_gate = None
        corrected_hidden = None
        if self.correction_head is not None:
            if self.config.correction_hidden_feedback:
                with torch.no_grad():
                    prev_gt_emb = self.embed_tokens(correction_previous_ids)
                previous_target_logits = None
                previous_target_mask = None
                previous_rank_features = None
                if correction_output_mode == "logits":
                    if selector_previous_logits_mask is not None:
                        previous_rank_features = selector_previous_rank_features
                        previous_target_mask = selector_previous_logits_mask
                    else:
                        target_blocks = targets.view(num_blocks, block, -1)
                        previous_target_logits = torch.cat(
                            [
                                torch.zeros_like(target_blocks[:, :1]),
                                target_blocks[:, :-1],
                            ],
                            dim=1,
                        )
                        previous_target_mask = block_positions > 0
                logits_blocks, correction_states, corrected_hidden = (
                    self._teacher_forced_hidden_feedback_correction(
                        hidden_blocks,
                        prev_gt_emb,
                        block_positions,
                        base_logits_blocks,
                        previous_target_logits,
                        previous_target_mask,
                        previous_rank_features=previous_rank_features,
                        current_token_embeddings=selector_current_embeddings,
                    )
                )
                logits = logits_blocks.reshape(1, mask_tokens_size, -1)
            elif self.config.sample_from_anchor:
                with torch.no_grad():
                    prev_gt_emb = self.embed_tokens(correction_previous_ids)
                if correction_output_mode == "logits":
                    if (
                        base_logits_blocks is None
                        and not self.config.correction_project_corrected_hidden
                    ):
                        raise RuntimeError(
                            "Logit-residual Correction requires base logits"
                        )
                    previous_feature_kwargs = {}
                    if selector_previous_logits_mask is not None:
                        previous_target_logits = None
                        previous_target_mask = selector_previous_logits_mask
                        if selector_previous_rank_features is not None:
                            previous_feature_kwargs["previous_rank_features"] = (
                                selector_previous_rank_features
                            )
                    else:
                        target_blocks = targets.view(num_blocks, block, -1)
                        previous_target_logits = torch.cat(
                            [
                                torch.zeros_like(target_blocks[:, :1]),
                                target_blocks[:, :-1],
                            ],
                            dim=1,
                        )
                        previous_target_mask = block_positions > 0
                    delta_logits, correction_states, _ = self.correction_head(
                        prev_gt_emb,
                        hidden_blocks,
                        block_positions,
                        previous_logits=previous_target_logits,
                        previous_logits_mask=previous_target_mask,
                        **selector_current_kwargs,
                        **previous_feature_kwargs,
                    )
                    if (
                        self.config.correction_project_corrected_hidden
                        or self.config.correction_hidden_aux_loss
                    ):
                        delta_hidden = (
                            self.correction_head.auxiliary_hidden_residual(
                                correction_states,
                            )
                        )
                        corrected_hidden = hidden_blocks + delta_hidden.to(
                            hidden_blocks.dtype
                        )
                    if self.config.correction_project_corrected_hidden:
                        projected_logits = self.lm_head(
                            corrected_hidden.reshape(1, mask_tokens_size, -1).to(
                                self.lm_head.weight.dtype
                            )
                        )
                        logits = projected_logits + delta_logits.reshape(
                            1, mask_tokens_size, -1
                        ).to(projected_logits.dtype)
                    else:
                        assert base_logits_blocks is not None  # noqa: S101
                        logits = (
                            base_logits_blocks
                            + delta_logits.to(base_logits_blocks.dtype)
                        ).reshape(1, mask_tokens_size, -1)
                else:
                    delta_hidden, correction_states, _ = self.correction_head(
                        prev_gt_emb,
                        hidden_blocks,
                        block_positions,
                        **selector_current_kwargs,
                    )
                    corrected_hidden = hidden_blocks + delta_hidden.to(
                        hidden_blocks.dtype
                    )
            else:
                with torch.no_grad():
                    prev_gt_emb = self.embed_tokens(correction_previous_ids[:, 1:])
                if correction_output_mode == "logits":
                    if (
                        base_logits_blocks is None
                        and not self.config.correction_project_corrected_hidden
                    ):
                        raise RuntimeError(
                            "Logit-residual Correction requires base logits"
                        )
                    previous_feature_kwargs = {}
                    if selector_previous_logits_mask is not None:
                        previous_target_logits = None
                        previous_target_mask = selector_previous_logits_mask[:, 1:]
                        if selector_previous_rank_features is not None:
                            previous_feature_kwargs["previous_rank_features"] = (
                                selector_previous_rank_features[:, 1:]
                            )
                    else:
                        target_blocks = targets.view(num_blocks, block, -1)
                        previous_target_logits = target_blocks[:, :-1]
                        previous_target_mask = torch.ones(
                            num_blocks,
                            block - 1,
                            dtype=torch.bool,
                            device=hidden.device,
                        )
                    delta_logits, draft_states, _ = self.correction_head(
                        prev_gt_emb,
                        hidden_blocks[:, 1:],
                        block_positions[:, 1:],
                        previous_logits=previous_target_logits,
                        previous_logits_mask=previous_target_mask,
                        **selector_current_tail_kwargs,
                        **previous_feature_kwargs,
                    )
                    if (
                        self.config.correction_project_corrected_hidden
                        or self.config.correction_hidden_aux_loss
                    ):
                        delta_hidden = (
                            self.correction_head.auxiliary_hidden_residual(
                                draft_states,
                            )
                        )
                        corrected_hidden = torch.cat(
                            [
                                hidden_blocks[:, :1],
                                hidden_blocks[:, 1:]
                                + delta_hidden.to(hidden_blocks.dtype),
                            ],
                            dim=1,
                        )
                    if self.config.correction_project_corrected_hidden:
                        projected_logits = self.lm_head(
                            corrected_hidden.reshape(1, mask_tokens_size, -1).to(
                                self.lm_head.weight.dtype
                            )
                        )
                        full_delta_logits = torch.cat(
                            [
                                delta_logits.new_zeros(
                                    num_blocks, 1, delta_logits.shape[-1]
                                ),
                                delta_logits,
                            ],
                            dim=1,
                        )
                        logits = projected_logits + full_delta_logits.reshape(
                            1, mask_tokens_size, -1
                        ).to(projected_logits.dtype)
                    else:
                        assert base_logits_blocks is not None  # noqa: S101
                        logits_blocks = torch.cat(
                            [
                                base_logits_blocks[:, :1],
                                base_logits_blocks[:, 1:]
                                + delta_logits.to(base_logits_blocks.dtype),
                            ],
                            dim=1,
                        )
                        logits = logits_blocks.reshape(1, mask_tokens_size, -1)
                else:
                    delta_hidden, draft_states, _ = self.correction_head(
                        prev_gt_emb,
                        hidden_blocks[:, 1:],
                        block_positions[:, 1:],
                        **selector_current_tail_kwargs,
                    )
                    corrected_hidden = torch.cat(
                        [
                            hidden_blocks[:, :1],
                            hidden_blocks[:, 1:]
                            + delta_hidden.to(hidden_blocks.dtype),
                        ],
                        dim=1,
                    )
                correction_states = torch.cat(
                    [
                        draft_states.new_zeros(
                            num_blocks, 1, draft_states.shape[-1]
                        ),
                        draft_states,
                    ],
                    dim=1,
                )

            if (
                correction_output_mode == "hidden"
                and not self.config.correction_hidden_feedback
            ):
                # Hidden mode projects the corrected block once.
                logits = self.lm_head(
                    corrected_hidden.reshape(1, mask_tokens_size, -1).to(
                        self.lm_head.weight.dtype
                    )
                )
            if self.markov_head is not None:
                collaboration_base_logits = logits
                collaborative_blocks, collaboration_gate, prev_emb = (
                    self._apply_collaborative_markov(
                        logits.view(num_blocks, block, -1),
                        correction_states,
                        correction_previous_ids,
                        hidden_blocks,
                    )
                )
                logits = collaborative_blocks.reshape(1, mask_tokens_size, -1)

            # Optional validation-only base projection for change/gain diagnostics.
            # It is never part of the training or inference correction path.
            if not self.training and self.config.correction_base_diagnostics:
                if base_logits is None:
                    with torch.no_grad():
                        base_logits = self.lm_head(
                            hidden.detach().to(self.lm_head.weight.dtype)
                        )

            # Validation keeps the teacher-forced view for comparison and also
            # measures the actual autoregressive feedback chain.
            if not self.training and self.config.correction_rollout_metrics:
                rollout_initial_logits = None
                if (
                    not self.config.sample_from_anchor
                    and correction_output_mode == "logits"
                ):
                    rollout_initial_logits = targets.view(num_blocks, block, -1)[:, 0]
                _, rollout_blocks = self.rollout_correction(
                    hidden_blocks.detach(),
                    anchor_token_ids=block_tokens[:, 0],
                    initial_previous_logits=rollout_initial_logits,
                    base_logits=base_logits_blocks,
                )
                rollout_logits = rollout_blocks.reshape(1, mask_tokens_size, -1)
        elif self.markov_head is not None:
            if logits is None:
                raise RuntimeError("Markov correction requires base logits")
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
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
                    hidden_blocks,
                    block_tokens[:, 0],
                    aligned_loss_mask,
                    teacher_previous_token_ids=prev_token_ids,
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
                hidden_blocks,
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
        )
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
                aligned_loss_mask.view(num_blocks, block),
            )
            loss = loss + (self.config.correction_hidden_aux_weight * hidden_aux_loss)
            metrics["loss_sum"] = loss.detach().clone()
            metrics["correction_hidden_aux_loss_sum"] = hidden_aux_loss.detach().clone()
            metrics["correction_hidden_aux_loss_total"] = torch.ones(
                (),
                device=loss.device,
                dtype=torch.float32,
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
                (
                    conditioning_current_ids,
                    conditioning_previous_ids,
                    conditioning_previous_rank_features,
                    conditioning_previous_logits_mask,
                ) = self._selector_correction_inputs(
                    candidate_ids,
                    selector_logits,
                    selected_ids,
                    anchor_token_ids,
                    initial_previous_logits=initial_previous_logits,
                )
        tokens, logits, _, _ = self._rollout_correction_steps(
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
        return tokens, logits
