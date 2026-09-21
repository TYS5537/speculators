"""Baseline DSpark: DFlash with Markov and confidence heads."""

from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics, select_logged_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.metrics import LossConfig, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

_DSPARK_PAPER_LOSS_FN = '{"ce": 0.1, "tv": 0.9}'
_DEFAULT_LOSS_CONFIG: LossConfig = resolve_loss_config(_DSPARK_PAPER_LOSS_FN)

__all__ = ["DSparkDraftModel"]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """The baseline sequential Markov drafter; experimental heads live in MUSE."""

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)
        self._init_sequential_heads(config)

    def _init_sequential_heads(self, config: DSparkSpeculatorConfig) -> None:
        """Initialize the baseline heads; MUSE overrides this extension point."""
        hidden_size = config.transformer_layer_config.hidden_size
        self.markov_head: MarkovHead | None = None
        # Kept as an empty capability for shared evaluators, not an architecture.
        self.correction_head = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )
        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires an enabled Markov head."
                )
            sequential_dim = (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(hidden_size + sequential_dim)

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
        config = cls.config_class(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
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
            **kwargs,
        )
        if logits is None:
            raise RuntimeError("DSpark forward requires projected draft logits")
        num_blocks = hidden.shape[1] // self.block_size
        block = self.block_size
        mask_tokens_size = num_blocks * block
        base_logits = logits
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        prev_token_ids = (
            block_tokens
            if self.config.sample_from_anchor
            else torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
        )
        hidden_blocks = hidden.view(num_blocks, block, -1)
        prev_emb = None
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )
        confidence_logits = None
        if self.confidence_head is not None:
            features = self._confidence_features(
                hidden_blocks,
                prev_emb if self.config.confidence_head_with_markov else None,
                detach=self.config.confidence_detach_features,
            )
            confidence_logits = self.confidence_head(features).reshape(
                1, mask_tokens_size
            )
        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            block,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            confidence_length_alpha=confidence_length_alpha,
            confidence_loss_weighting=confidence_loss_weighting,
            first_error_focal_alpha=first_error_focal_alpha,
            adaptive_loss=adaptive_loss,
            ssal_decay_weight=ssal_decay_weight,
            base_logits=base_logits if self.markov_head is not None else None,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
            target_log_normalizer=target_log_normalizer,
            target_argmax_ids=target_argmax_ids,
        )
        return None, loss, select_logged_metrics(metrics)
