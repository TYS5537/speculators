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
    representation-level supervision and recurrence. An optional verifier-confirmed
    gated memory carries information between blocks. The confidence head predicts
    each position's acceptance probability.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size
        if (
            config.correction_generated_token_ratio > 0.0
            and not config.enable_correction_head
        ):
            raise ValueError("correction_generated_token_ratio > 0 requires Correction")
        if (
            config.correction_output_mode != "hidden"
            and not config.enable_correction_head
        ):
            raise ValueError("correction_output_mode='logits' requires Correction")
        if config.correction_lm_head_fusion:
            if not config.enable_correction_head:
                raise ValueError("correction_lm_head_fusion=True requires Correction")
            if config.correction_output_mode != "hidden":
                raise ValueError(
                    "correction_lm_head_fusion=True requires hidden output mode"
                )
        if config.correction_moe and not config.enable_correction_head:
            raise ValueError("correction_moe=True requires Correction")
        if config.correction_moe_logit_routing and not config.correction_moe:
            raise ValueError("correction_moe_logit_routing=True requires MoE")
        if (
            config.correction_hidden_aux_loss
            or config.correction_hidden_feedback
            or config.correction_cross_block_memory
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
        if (
            config.correction_generated_token_warmup
            + config.correction_generated_token_ramp
            > 1.0
        ):
            raise ValueError("Correction generated-token warmup + ramp must be <= 1")

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
                block_memory_size=(
                    config.correction_hidden_size
                    if config.correction_cross_block_memory
                    else None
                ),
                enable_moe=config.correction_moe,
                moe_shared_rank=config.correction_moe_shared_rank,
                moe_expert_rank=config.correction_moe_expert_rank,
                moe_num_experts=config.correction_moe_num_experts,
                moe_logit_routing=config.correction_moe_logit_routing,
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

        self.cross_block_memory_verifier_proj: torch.nn.Linear | None = None
        self.cross_block_memory_token_proj: torch.nn.Linear | None = None
        self.cross_block_memory_gate: torch.nn.Linear | None = None
        if config.correction_cross_block_memory:
            memory_size = config.correction_hidden_size
            self.cross_block_memory_verifier_proj = torch.nn.Linear(
                hidden_size, memory_size, bias=False
            )
            self.cross_block_memory_token_proj = torch.nn.Linear(
                hidden_size, memory_size, bias=False
            )
            self.cross_block_memory_gate = torch.nn.Linear(memory_size, 1)
            torch.nn.init.zeros_(self.cross_block_memory_gate.weight)
            torch.nn.init.constant_(
                self.cross_block_memory_gate.bias,
                config.correction_memory_gate_bias,
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
            correction_moe=kwargs.get("correction_moe", False),
            correction_moe_shared_rank=kwargs.get("correction_moe_shared_rank", 128),
            correction_moe_expert_rank=kwargs.get("correction_moe_expert_rank", 64),
            correction_moe_num_experts=kwargs.get("correction_moe_num_experts", 4),
            correction_moe_load_balance_weight=kwargs.get(
                "correction_moe_load_balance_weight", 0.01
            ),
            correction_moe_logit_routing=kwargs.get(
                "correction_moe_logit_routing", False
            ),
            correction_hidden_aux_loss=kwargs.get("correction_hidden_aux_loss", False),
            correction_hidden_aux_weight=kwargs.get(
                "correction_hidden_aux_weight", 0.1
            ),
            correction_hidden_feedback=kwargs.get("correction_hidden_feedback", False),
            correction_cross_block_memory=kwargs.get(
                "correction_cross_block_memory", False
            ),
            correction_memory_gate_bias=kwargs.get("correction_memory_gate_bias", -2.0),
            correction_project_corrected_hidden=kwargs.get(
                "correction_project_corrected_hidden", False
            ),
            correction_with_markov=kwargs.get("correction_with_markov", False),
            correction_markov_gate_bias=kwargs.get("correction_markov_gate_bias", -2.0),
            correction_generated_token_ratio=kwargs.get(
                "correction_generated_token_ratio", 0.0
            ),
            correction_generated_token_warmup=kwargs.get(
                "correction_generated_token_warmup", 0.2
            ),
            correction_generated_token_ramp=kwargs.get(
                "correction_generated_token_ramp", 0.4
            ),
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
        generated_token_ratio = float(
            kwargs.get("correction_generated_token_ratio", 0.0)
        )
        generated_token_warmup = float(
            kwargs.get("correction_generated_token_warmup", 0.2)
        )
        generated_token_ramp = float(kwargs.get("correction_generated_token_ramp", 0.4))
        if not 0.0 <= generated_token_ratio <= 1.0:
            raise ValueError("Generated-token ratio must be in [0, 1]")
        if not 0.0 <= generated_token_warmup <= 1.0:
            raise ValueError("Generated-token warmup must be in [0, 1]")
        if not 0.0 <= generated_token_ramp <= 1.0:
            raise ValueError("Generated-token ramp must be in [0, 1]")
        if generated_token_warmup + generated_token_ramp > 1.0:
            raise ValueError("Generated-token warmup + ramp must be <= 1")
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
        if generated_token_ratio > 0.0:
            train_kw["correction_generated_token_curriculum"] = True
            train_kw["correction_generated_token_target_ratio"] = generated_token_ratio
            train_kw["correction_generated_token_warmup"] = generated_token_warmup
            train_kw["correction_generated_token_ramp"] = generated_token_ramp
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

    def _cross_block_memory_features(
        self,
        verifier_pre_lm_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a verifier/token memory candidate and its scalar update gate."""
        if (
            self.cross_block_memory_verifier_proj is None
            or self.cross_block_memory_token_proj is None
            or self.cross_block_memory_gate is None
        ):
            raise RuntimeError("Correction cross-block memory is not enabled")
        if verifier_pre_lm_hidden.ndim != 2:
            raise ValueError("verifier_pre_lm_hidden must be rank-2")
        if anchor_token_ids.shape != verifier_pre_lm_hidden.shape[:1]:
            raise ValueError("anchor token IDs and verifier pre-LM hidden must align")

        dtype = self.cross_block_memory_verifier_proj.weight.dtype
        with torch.no_grad():
            anchor_embeddings = self.embed_tokens(anchor_token_ids.long())
        candidate = torch.tanh(
            self.cross_block_memory_verifier_proj(
                verifier_pre_lm_hidden.detach().to(dtype)
            )
            + self.cross_block_memory_token_proj(anchor_embeddings.to(dtype))
        )
        update_gate = torch.sigmoid(self.cross_block_memory_gate(candidate))
        return candidate, update_gate

    def update_cross_block_memory(
        self,
        previous_memory: torch.Tensor | None,
        verifier_pre_lm_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Update memory after verification from committed context only.

        ``verifier_pre_lm_hidden`` represents the last processed, committed token;
        ``anchor_token_ids`` is the token that anchors the next proposal.
        """
        candidate, update_gate = self._cross_block_memory_features(
            verifier_pre_lm_hidden,
            anchor_token_ids,
        )
        if previous_memory is None:
            previous_memory = torch.zeros_like(candidate)
        if previous_memory.shape != candidate.shape:
            raise ValueError(
                "Previous cross-block memory and memory candidate must align"
            )
        previous_memory = previous_memory.to(candidate.dtype)
        return previous_memory + update_gate * (candidate - previous_memory)

    @torch.compiler.disable
    def _teacher_forced_cross_block_memory(
        self,
        verifier_last_hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        document_ids: torch.Tensor,
        block_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Build causal block memories from the verifier-confirmed GT sequence."""
        if (
            anchor_positions.ndim != 1
            or anchor_token_ids.shape != anchor_positions.shape
        ):
            raise ValueError("Anchor positions and token IDs must be rank-1 and align")
        if block_valid.shape != anchor_positions.shape:
            raise ValueError("Block validity mask and anchor positions must align")

        context_positions = (anchor_positions - 1).clamp_min(0)
        with torch.no_grad():
            verifier_pre_lm_hidden = self.verifier_norm(
                verifier_last_hidden_states[:, context_positions, :].to(
                    self.verifier_norm.weight.dtype
                )
            )[0]
        candidates, update_gates = self._cross_block_memory_features(
            verifier_pre_lm_hidden,
            anchor_token_ids,
        )

        anchor_docs = document_ids[0, anchor_positions]
        context_docs = document_ids[0, context_positions]
        valid_context = (
            block_valid.bool()
            & (anchor_positions > 0)
            & (anchor_docs == context_docs)
            & (anchor_docs != -1)
        )

        memory = torch.zeros_like(candidates[:1])
        memories: list[torch.Tensor] = []
        previous_valid = torch.zeros(
            (), dtype=torch.bool, device=anchor_positions.device
        )
        previous_doc = anchor_docs.new_full((), -1)
        previous_anchor = anchor_positions.new_full((), -1)
        for block_idx in range(anchor_positions.numel()):
            continues_document = (
                previous_valid
                & valid_context[block_idx]
                & (anchor_docs[block_idx] == previous_doc)
                & (anchor_positions[block_idx] > previous_anchor)
            )
            memory = memory * continues_document.to(memory.dtype)
            memory = memory + update_gates[block_idx : block_idx + 1] * (
                candidates[block_idx : block_idx + 1] - memory
            )
            memory = memory * valid_context[block_idx].to(memory.dtype)
            memories.append(memory[0])
            previous_valid = valid_context[block_idx]
            previous_doc = anchor_docs[block_idx]
            previous_anchor = anchor_positions[block_idx]
        return torch.stack(memories, dim=0)

    @torch.compiler.disable
    def _teacher_forced_hidden_feedback_correction(
        self,
        dflash_hidden: torch.Tensor,
        previous_token_embeddings: torch.Tensor,
        block_positions: torch.Tensor,
        base_logits: torch.Tensor | None,
        previous_target_logits: torch.Tensor | None,
        previous_target_logits_mask: torch.Tensor | None,
        block_memory: torch.Tensor | None = None,
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
                if block_memory is not None:
                    head_kwargs["block_memory"] = block_memory
                needs_previous_logits = (
                    self.correction_head.output_mode == "logits"
                    or self.config.correction_moe_logit_routing
                )
                if needs_previous_logits:
                    if (
                        previous_target_logits is None
                        or previous_target_logits_mask is None
                    ):
                        raise RuntimeError(
                            "Logit-aware Correction requires previous target logits"
                        )
                    head_kwargs["previous_logits"] = previous_target_logits[
                        :, position : position + 1
                    ]
                    head_kwargs["previous_logits_mask"] = previous_target_logits_mask[
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
                        previous_logits=head_kwargs.get("previous_logits"),
                        previous_logits_mask=head_kwargs.get("previous_logits_mask"),
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

    @torch.compiler.disable
    def _generated_feedback_correction(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        block_memory: torch.Tensor | None = None,
        initial_previous_logits: torch.Tensor | None = None,
        return_selector_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a differentiable Correction pass with greedy token self-feedback.

        Token selection is discrete, but the per-position logits and causal K/V
        states retain their autograd graph.  The first input is always the real
        anchor; every later token input is generated by the current Correction
        model. Logit-residual mode also feeds back the previous final logits,
        detached across the discrete autoregressive boundary.
        """
        if self.correction_head is None:
            raise RuntimeError(
                "_generated_feedback_correction requires enable_correction_head=True"
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
        logit_routing_enabled = getattr(
            self.config, "correction_moe_logit_routing", False
        )
        logit_feedback_enabled = (
            correction_output_mode == "logits" or logit_routing_enabled
        )
        fused_lm_head_enabled = (
            bool(getattr(self.config, "correction_lm_head_fusion", False))
            and correction_output_mode == "hidden"
            and not torch.is_grad_enabled()
        )
        fused_base_logits = None
        if fused_lm_head_enabled:
            # Project the full parallel DFlash block once. Each sequential
            # Correction position adds an inference-cached rank-to-vocabulary
            # residual instead of invoking the full LM head again.
            fused_base_logits = self.lm_head(
                dflash_hidden.to(self.lm_head.weight.dtype)
            )
        previous_feedback_logits = None
        previous_feedback_mask = None
        if logit_feedback_enabled:
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
        previous_corrected_hidden = None
        previous_corrected_hidden_mask = None
        block_memory_kwargs = (
            {} if block_memory is None else {"block_memory": block_memory}
        )
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
                if fused_base_logits is not None:
                    final_logits = fused_base_logits[:, position]
                else:
                    final_logits = self.lm_head(
                        current_hidden.to(self.lm_head.weight.dtype)
                    )
                causal_states = current_hidden.new_zeros(
                    current_hidden.shape[0], self.config.correction_hidden_size
                )
            else:
                with torch.no_grad():
                    previous_emb = self.embed_tokens(previous_ids).unsqueeze(1)
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
                        **block_memory_kwargs,
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
                            previous_logits=previous_feedback_logits,
                            previous_logits_mask=previous_feedback_mask,
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
                        **block_memory_kwargs,
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
                            previous_logits=previous_feedback_logits,
                            previous_logits_mask=previous_feedback_mask,
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
                        previous_ids.unsqueeze(1),
                        dflash_hidden[:, position : position + 1],
                    )
                    final_logits = final_logits[:, 0]

            proposal_logits = final_logits
            candidate_ids = None
            if (
                getattr(self, "candidate_selector", None) is not None
                and position >= start_position
            ):
                candidate_ids, candidate_logits = self.dflash2_select_candidates(
                    final_logits,
                    corrected_current_hidden,
                    previous_ids,
                )
                if return_selector_logits:
                    proposal_logits = torch.full_like(final_logits, -torch.inf)
                    proposal_logits.scatter_(-1, candidate_ids, candidate_logits)
                sampling_logits = candidate_logits
            else:
                sampling_logits = final_logits
            with torch.no_grad():
                if temperature > 0:
                    probabilities = torch.softmax(
                        sampling_logits.float() / temperature, dim=-1
                    )
                    sampled_indices = torch.multinomial(
                        probabilities, num_samples=1
                    ).squeeze(-1)
                else:
                    sampled_indices = torch.argmax(sampling_logits, dim=-1)
                draft_ids = sampled_indices
                if candidate_ids is not None:
                    draft_ids = candidate_ids.gather(
                        -1, sampled_indices.unsqueeze(-1)
                    ).squeeze(-1)
            output_tokens.append(draft_ids)
            output_logits.append(proposal_logits)
            output_states.append(causal_states)
            output_corrected_hidden.append(corrected_current_hidden)

            if logit_feedback_enabled and position >= start_position:
                previous_feedback_logits = final_logits.detach().unsqueeze(1)
                previous_feedback_mask = torch.ones(
                    final_logits.shape[0],
                    1,
                    dtype=torch.bool,
                    device=final_logits.device,
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
            previous_ids = draft_ids
            if self.d2t is not None:
                previous_ids = previous_ids + self.d2t[previous_ids]

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
        correction_use_generated_tokens: bool = False,
        correction_generated_token_ratio: torch.Tensor | float = 0.0,
        correction_generated_token_curriculum_active: bool = False,
        **kwargs,
    ):
        correction_output_mode = (
            getattr(self.correction_head, "output_mode", "hidden")
            if self.correction_head is not None
            else None
        )
        generated_correction_training = (
            self.training
            and self.correction_head is not None
            and correction_use_generated_tokens
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
                    or (
                        correction_output_mode == "logits"
                        and not generated_correction_training
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
        block_memory = None
        if self.config.correction_cross_block_memory:
            anchor_positions = anchored_block_indices.view(num_blocks, block)[:, 0]
            block_valid = aligned_loss_mask.view(num_blocks, block).bool().any(dim=1)
            block_memory = self._teacher_forced_cross_block_memory(
                verifier_last_hidden_states,
                block_tokens[:, 0],
                anchor_positions,
                document_ids,
                block_valid,
            )

        confidence_logits = None
        prev_emb = None
        correction_states = None
        rollout_logits = None
        collaboration_base_logits = None
        collaboration_gate = None
        corrected_hidden = None
        generated_correction_tokens = None
        moe_previous_logits = None
        moe_previous_logits_mask = None
        if self.config.correction_moe_logit_routing:
            target_blocks = targets.view(num_blocks, block, -1)
            moe_previous_logits = torch.cat(
                [
                    torch.zeros_like(target_blocks[:, :1]),
                    target_blocks[:, :-1],
                ],
                dim=1,
            )
            moe_previous_logits_mask = block_positions > 0
        if self.correction_head is not None:
            if generated_correction_training:
                initial_previous_logits = None
                if not self.config.sample_from_anchor and (
                    correction_output_mode == "logits"
                    or self.config.correction_moe_logit_routing
                ):
                    initial_previous_logits = targets.view(num_blocks, block, -1)[:, 0]
                (
                    generated_correction_tokens,
                    logits_blocks,
                    correction_states,
                    corrected_hidden,
                ) = (
                    self._generated_feedback_correction(
                        hidden_blocks,
                        anchor_token_ids=block_tokens[:, 0],
                        block_memory=block_memory,
                        initial_previous_logits=initial_previous_logits,
                    )
                )
                logits = logits_blocks.reshape(1, mask_tokens_size, -1)
                if self.config.correction_moe_logit_routing:
                    moe_previous_logits = torch.cat(
                        [
                            torch.zeros_like(logits_blocks[:, :1]),
                            logits_blocks[:, :-1].detach(),
                        ],
                        dim=1,
                    )
            else:
                if self.config.correction_hidden_feedback:
                    with torch.no_grad():
                        prev_gt_emb = self.embed_tokens(prev_token_ids)
                    previous_target_logits = None
                    previous_target_mask = None
                    if (
                        correction_output_mode == "logits"
                        or self.config.correction_moe_logit_routing
                    ):
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
                            block_memory,
                        )
                    )
                    logits = logits_blocks.reshape(1, mask_tokens_size, -1)
                elif self.config.sample_from_anchor:
                    with torch.no_grad():
                        prev_gt_emb = self.embed_tokens(block_tokens)
                    if correction_output_mode == "logits":
                        if (
                            base_logits_blocks is None
                            and not self.config.correction_project_corrected_hidden
                        ):
                            raise RuntimeError(
                                "Logit-residual Correction requires base logits"
                            )
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
                            block_memory=block_memory,
                        )
                        if (
                            self.config.correction_project_corrected_hidden
                            or self.config.correction_hidden_aux_loss
                        ):
                            delta_hidden = (
                                self.correction_head.auxiliary_hidden_residual(
                                    correction_states,
                                    previous_logits=previous_target_logits,
                                    previous_logits_mask=previous_target_mask,
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
                        logit_routing_kwargs = {}
                        if self.config.correction_moe_logit_routing:
                            logit_routing_kwargs = {
                                "previous_logits": moe_previous_logits,
                                "previous_logits_mask": (moe_previous_logits_mask),
                            }
                        delta_hidden, correction_states, _ = self.correction_head(
                            prev_gt_emb,
                            hidden_blocks,
                            block_positions,
                            block_memory=block_memory,
                            **logit_routing_kwargs,
                        )
                        corrected_hidden = hidden_blocks + delta_hidden.to(
                            hidden_blocks.dtype
                        )
                else:
                    with torch.no_grad():
                        prev_gt_emb = self.embed_tokens(block_tokens[:, :-1])
                    if correction_output_mode == "logits":
                        if (
                            base_logits_blocks is None
                            and not self.config.correction_project_corrected_hidden
                        ):
                            raise RuntimeError(
                                "Logit-residual Correction requires base logits"
                            )
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
                            block_memory=block_memory,
                        )
                        if (
                            self.config.correction_project_corrected_hidden
                            or self.config.correction_hidden_aux_loss
                        ):
                            delta_hidden = (
                                self.correction_head.auxiliary_hidden_residual(
                                    draft_states,
                                    previous_logits=previous_target_logits,
                                    previous_logits_mask=previous_target_mask,
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
                        logit_routing_kwargs = {}
                        if self.config.correction_moe_logit_routing:
                            assert moe_previous_logits is not None  # noqa: S101
                            assert moe_previous_logits_mask is not None  # noqa: S101
                            logit_routing_kwargs = {
                                "previous_logits": moe_previous_logits[:, 1:],
                                "previous_logits_mask": (
                                    moe_previous_logits_mask[:, 1:]
                                ),
                            }
                        delta_hidden, draft_states, _ = self.correction_head(
                            prev_gt_emb,
                            hidden_blocks[:, 1:],
                            block_positions[:, 1:],
                            block_memory=block_memory,
                            **logit_routing_kwargs,
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
                    # Hidden mode projects the corrected block once. Generated-token
                    # training projects each position inside its feedback loop.
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
                            prev_token_ids,
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
            # measures the actual generated-token feedback chain.
            if not self.training and self.config.correction_rollout_metrics:
                rollout_initial_logits = None
                if not self.config.sample_from_anchor and (
                    correction_output_mode == "logits"
                    or self.config.correction_moe_logit_routing
                ):
                    rollout_initial_logits = targets.view(num_blocks, block, -1)[:, 0]
                _, rollout_blocks = self.rollout_correction(
                    hidden_blocks.detach(),
                    anchor_token_ids=block_tokens[:, 0],
                    block_memory=block_memory,
                    initial_previous_logits=rollout_initial_logits,
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
        selector_loss = None
        if self.candidate_selector is not None:
            selector_hidden = (
                corrected_hidden if corrected_hidden is not None else hidden_blocks
            )
            realized_previous_token_ids = None
            if generated_correction_tokens is not None:
                start_position = 0 if self.config.sample_from_anchor else 1
                realized_previous_token_ids = prev_token_ids.clone()
                realized_previous_token_ids[:, start_position] = block_tokens[:, 0]
                if start_position + 1 < block:
                    realized_previous_token_ids[:, start_position + 1 :] = (
                        self._draft_ids_to_verifier(
                            generated_correction_tokens[:, start_position:-1]
                        )
                    )
            elif self.correction_head is not None or self.markov_head is not None:
                # These sequential heads were teacher-forced above. Their
                # downstream hidden/logits are only valid for the GT predecessor.
                realized_previous_token_ids = prev_token_ids
            teacher_previous_token_ids = (
                realized_previous_token_ids
                if realized_previous_token_ids is not None
                else prev_token_ids
            )
            candidate_ids, candidate_logits, selector_loss = (
                self._dflash2_block_outputs(
                    logits,
                    targets,
                    selector_hidden,
                    block_tokens[:, 0],
                    aligned_loss_mask,
                    teacher_previous_token_ids=teacher_previous_token_ids,
                    realized_previous_token_ids=realized_previous_token_ids,
                )
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
        if self.config.correction_moe:
            if self.correction_head is None or correction_states is None:
                raise RuntimeError("Correction MoE requires causal states")
            moe_balance_loss, moe_router_entropy = (
                self.correction_head.moe_router_statistics(
                    correction_states,
                    aligned_loss_mask.view(num_blocks, block).bool(),
                    previous_logits=moe_previous_logits,
                    previous_logits_mask=moe_previous_logits_mask,
                )
            )
            loss = loss + (
                self.config.correction_moe_load_balance_weight
                * moe_balance_loss.to(loss.dtype)
            )
            metrics["loss_sum"] = loss.detach().clone()
            one = torch.ones((), device=loss.device, dtype=torch.float32)
            metrics["correction_moe_balance_loss_sum"] = (
                moe_balance_loss.detach().clone()
            )
            metrics["correction_moe_balance_loss_total"] = one
            metrics["correction_moe_router_entropy_sum"] = (
                moe_router_entropy.detach().clone()
            )
            metrics["correction_moe_router_entropy_total"] = one.clone()
        metrics = select_logged_metrics(
            metrics,
            include_diagnostics=(
                not self.training and self.config.correction_base_diagnostics
            ),
        )
        if (
            self.training
            and self.correction_head is not None
            and correction_generated_token_curriculum_active
        ):
            ratio = torch.as_tensor(
                correction_generated_token_ratio,
                device=loss.device,
                dtype=torch.float32,
            ).detach()
            one = torch.ones((), device=loss.device, dtype=torch.float32)
            metrics["correction_generated_token_ratio_sum"] = ratio
            metrics["correction_generated_token_ratio_total"] = one
        return None, loss, metrics

    @torch.compiler.disable
    @torch.no_grad()
    def rollout_correction(
        self,
        dflash_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        *,
        temperature: float = 0.0,
        block_memory: torch.Tensor | None = None,
        initial_previous_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively apply correction using generated-token feedback."""
        tokens, logits, _, _ = self._generated_feedback_correction(
            dflash_hidden,
            anchor_token_ids,
            temperature=temperature,
            block_memory=block_memory,
            initial_previous_logits=initial_previous_logits,
            return_selector_logits=True,
        )
        return tokens, logits
