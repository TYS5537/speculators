from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import (
    CausalCorrectionHead,
    ConfidenceHead,
    MarkovHead,
)
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        self.correction_head: CausalCorrectionHead | None = None
        if config.enable_correction_head:
            self.correction_head = CausalCorrectionHead(
                input_hidden_size=hidden_size,
                token_embedding_size=hidden_size,
                correction_hidden_size=config.correction_hidden_size,
                correction_rank=config.correction_rank,
                draft_vocab_size=self.draft_vocab_size,
                num_layers=config.correction_num_layers,
                num_heads=config.correction_num_heads,
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
        enable_confidence_head_arg = kwargs.get("enable_confidence_head")
        confidence_head_with_markov_arg = kwargs.get("confidence_head_with_markov")
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_correction_head=kwargs.get("enable_correction_head", False),
            correction_hidden_size=kwargs.get("correction_hidden_size", 512),
            correction_rank=kwargs.get("correction_rank", 256),
            correction_num_layers=kwargs.get("correction_num_layers", 1),
            correction_num_heads=kwargs.get("correction_num_heads", 8),
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
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 3072)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        confidence_length_alpha = kwargs.get("confidence_length_alpha", 0.0)
        confidence_loss_weighting = kwargs.get(
            "confidence_loss_weighting", "uniform"
        )
        first_error_focal_alpha = kwargs.get("first_error_focal_alpha", 0.0)
        cat_mode = kwargs.get("cat_mode", "none")
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
            "cat_mode": cat_mode,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

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
        gamma: float = 4.0,
        max_anchors: int = 3072,
        confidence_head_alpha: float = 1.0,
        confidence_length_alpha: float = 0.0,
        confidence_loss_weighting: str = "uniform",
        first_error_focal_alpha: float = 0.0,
        cat_mode: str = "none",
        curriculum_base_weight: torch.Tensor | float = 0.0,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )

        # DSpark: add the active causal correction and predict confidence.
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
        base_logits_blocks = base_logits.view(num_blocks, block, -1)

        confidence_logits = None
        prev_emb = None
        correction_states = None
        if self.correction_head is not None:
            # Teacher forcing: only previous GT tokens shift. DFlash hidden/base
            # logits remain aligned with the current prediction positions.
            if self.config.sample_from_anchor:
                # Correct all slots; prev for slot k is block_tokens[:, k]
                # (mirrors Markov prev_token_ids under sample_from_anchor).
                with torch.no_grad():
                    prev_gt_emb = self.embed_tokens(block_tokens)
                correction, draft_states, _ = self.correction_head(
                    prev_gt_emb,
                    hidden_blocks,
                )
                logits_blocks = base_logits_blocks + correction
                logits = logits_blocks.reshape(1, mask_tokens_size, -1)
                correction_states = draft_states
            else:
                # slots-1: correct draft positions only; prev is the prior GT token.
                prev_gt_ids = block_tokens[:, :-1]
                with torch.no_grad():
                    prev_gt_emb = self.embed_tokens(prev_gt_ids)
                correction, draft_states, _ = self.correction_head(
                    prev_gt_emb,
                    hidden_blocks[:, 1:],
                )
                logits_blocks = torch.cat(
                    [
                        base_logits_blocks[:, :1],
                        base_logits_blocks[:, 1:] + correction,
                    ],
                    dim=1,
                )
                logits = logits_blocks.reshape(1, mask_tokens_size, -1)
                correction_states = torch.cat(
                    [
                        draft_states.new_zeros(num_blocks, 1, draft_states.shape[-1]),
                        draft_states,
                    ],
                    dim=1,
                )
        elif self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            if (
                self.config.confidence_head_with_markov
                and correction_states is not None
            ):
                conf_features = torch.cat(
                    [hidden_blocks, correction_states.to(hidden_blocks.dtype)], dim=-1
                )
            elif self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1
                )
            else:
                conf_features = hidden_blocks
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
            cat_mode=cat_mode,  # type: ignore[arg-type]
            base_logits=base_logits if self.correction_head is not None else None,
            curriculum_base_weight=curriculum_base_weight,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        draft_tokens = torch.argmax(logits, dim=-1)
        return draft_tokens, loss, metrics

    @torch.no_grad()
    def rollout_correction(
        self,
        base_logits: torch.Tensor,  # [N, block_size, draft_vocab]
        dflash_hidden: torch.Tensor,  # [N, block_size, hidden]
        anchor_token_ids: torch.Tensor,  # [N], verifier vocab
        *,
        temperature: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a corrected draft block with sequential token feedback.

        DFlash is not rerun here: callers provide its parallel hidden states and
        base logits. Returned tensors exclude anchor slot 0.
        """
        if self.correction_head is None:
            raise RuntimeError(
                "rollout_correction requires enable_correction_head=True"
            )
        expected_ndim = 3
        if base_logits.ndim != expected_ndim or dflash_hidden.ndim != expected_ndim:
            raise ValueError("base_logits and dflash_hidden must both be rank-3")
        if base_logits.shape[:2] != dflash_hidden.shape[:2]:
            raise ValueError("base_logits and dflash_hidden block shapes must match")
        if base_logits.shape[1] != self.block_size:
            raise ValueError(
                f"Expected block_size={self.block_size}, got {base_logits.shape[1]}"
            )
        if anchor_token_ids.shape != (base_logits.shape[0],):
            raise ValueError(
                f"Expected anchor_token_ids shape {(base_logits.shape[0],)}, "
                f"got {anchor_token_ids.shape}"
            )

        previous_ids = anchor_token_ids.long()
        cache = None
        output_tokens = []
        output_logits = []
        for position in range(1, self.block_size):
            previous_emb = self.embed_tokens(previous_ids).unsqueeze(1)
            correction, _, cache = self.correction_head(
                previous_emb,
                dflash_hidden[:, position : position + 1],
                cache=cache,
                use_cache=True,
            )
            final_logits = base_logits[:, position] + correction[:, 0]
            if temperature > 0:
                probabilities = torch.softmax(
                    final_logits.float() / temperature, dim=-1
                )
                draft_ids = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
            else:
                draft_ids = torch.argmax(final_logits, dim=-1)
            output_tokens.append(draft_ids)
            output_logits.append(final_logits)
            previous_ids = draft_ids
            if self.d2t is not None:
                previous_ids = previous_ids + self.d2t[previous_ids]

        return torch.stack(output_tokens, dim=1), torch.stack(output_logits, dim=1)
