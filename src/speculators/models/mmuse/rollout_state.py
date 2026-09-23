"""Ephemeral tensor feedback for one Correction rollout, never model state.

Sampling, token IDs and the attention cache stay in the model's loop. This state
only selects the current feedback inputs and replaces them after a sampled slot.
Dense logits are detached; corrected hidden states and compact encoding keep their
existing gradient paths. Replaced tensors are not kept in a history or snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from speculators.models.mmuse.correction import CausalCorrectionHead
    from speculators.models.mmuse.runtime_types import InitialLogitFeedback


@dataclass(slots=True, eq=False)
class RolloutFeedbackState:
    """Mutable, call-local references; no parameter registration or tensor copies."""

    dense_logits: torch.Tensor | None
    dense_mask: torch.Tensor | None
    online_rank_features: torch.Tensor | None
    online_mask: torch.Tensor | None
    corrected_hidden: torch.Tensor | None = None
    hidden_mask: torch.Tensor | None = None

    @classmethod
    def from_initial(
        cls,
        initial: InitialLogitFeedback,
        dflash_hidden: torch.Tensor,
        *,
        hidden_feedback_enabled: bool,
    ) -> RolloutFeedbackState:
        """Keep initial logit references and seed optional masked hidden feedback."""
        state = cls(
            initial.dense_logits,
            initial.dense_mask,
            initial.online_rank_features,
            initial.online_mask,
        )
        if hidden_feedback_enabled:
            state.corrected_hidden = dflash_hidden.new_zeros(
                dflash_hidden.shape[0], 1, dflash_hidden.shape[-1]
            )
            state.hidden_mask = torch.zeros(
                dflash_hidden.shape[0],
                1,
                dtype=torch.bool,
                device=dflash_hidden.device,
            )
        return state

    def correction_kwargs(
        self,
        position: int,
        *,
        hidden_feedback_enabled: bool,
        logit_feedback_enabled: bool,
        has_conditioning_features: bool,
        online_selector: bool,
        conditioning_previous_rank_features: torch.Tensor | None,
        conditioning_previous_logits_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Keep static > online > dense precedence and original keyword ordering."""
        kwargs = {}
        if hidden_feedback_enabled:
            assert self.corrected_hidden is not None  # noqa: S101
            assert self.hidden_mask is not None  # noqa: S101
            kwargs["previous_corrected_hidden"] = self.corrected_hidden
            kwargs["previous_corrected_hidden_mask"] = self.hidden_mask
        if logit_feedback_enabled:
            if has_conditioning_features:
                assert conditioning_previous_logits_mask is not None  # noqa: S101
                kwargs["previous_logits_mask"] = conditioning_previous_logits_mask[
                    :, position : position + 1
                ]
                if conditioning_previous_rank_features is not None:
                    kwargs["previous_rank_features"] = (
                        conditioning_previous_rank_features[:, position : position + 1]
                    )
            elif online_selector:
                assert self.online_mask is not None  # noqa: S101
                kwargs["previous_logits_mask"] = self.online_mask
                if self.online_rank_features is not None:
                    kwargs["previous_rank_features"] = self.online_rank_features
            else:
                assert self.dense_logits is not None  # noqa: S101
                assert self.dense_mask is not None  # noqa: S101
                kwargs["previous_logits"] = self.dense_logits
                kwargs["previous_logits_mask"] = self.dense_mask
        return kwargs

    def advance(
        self,
        final_logits: torch.Tensor,
        corrected_current_hidden: torch.Tensor,
        *,
        position: int,
        start_position: int,
        logit_feedback_enabled: bool,
        has_conditioning_features: bool,
        online_selector: bool,
        hidden_feedback_enabled: bool,
        correction_head: CausalCorrectionHead,
    ) -> None:
        """Update logit feedback only for active slots; hidden also sees the anchor.

        Online encoding runs in the caller's grad mode, including when static
        compact features take precedence at the next step. Do not skip that call
        or move it across sampling, hidden feedback or next-token mapping.
        """
        if (
            logit_feedback_enabled
            and not has_conditioning_features
            and not online_selector
            and position >= start_position
        ):
            self.dense_logits = final_logits.detach().unsqueeze(1)
            self.dense_mask = torch.ones(
                final_logits.shape[0], 1, dtype=torch.bool, device=final_logits.device
            )
        if online_selector and logit_feedback_enabled and position >= start_position:
            self.online_mask = torch.ones(
                final_logits.shape[0], 1, dtype=torch.bool, device=final_logits.device
            )
            self.online_rank_features = correction_head.encode_previous_distribution(
                self.online_mask,
                previous_logits=final_logits.detach().unsqueeze(1),
            )
        if hidden_feedback_enabled:
            self.corrected_hidden = corrected_current_hidden.unsqueeze(1)
            self.hidden_mask = torch.ones(
                corrected_current_hidden.shape[0],
                1,
                dtype=torch.bool,
                device=corrected_current_hidden.device,
            )
