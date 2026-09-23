"""Per-position conditioning for teacher-forced hidden-feedback Correction.

The model owns recurrence, cache handoff and final projection. This helper only
slices conditioning tensors: it neither shifts positions nor detaches features.
"""

import torch


def build_feedback_conditioning(
    position: int,
    *,
    output_mode: str,
    current_token_embeddings: torch.Tensor | None,
    previous_target_logits: torch.Tensor | None,
    previous_target_logits_mask: torch.Tensor | None,
    previous_rank_features: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Slice one active slot, preferring compact rank features over dense logits.

    Call only for active slots. A reserved anchor never validates or consumes
    these inputs; missing logit features fail before that slot's head call.
    """
    conditioning = {}
    if current_token_embeddings is not None:
        conditioning["current_token_embeddings"] = current_token_embeddings[
            :, position : position + 1
        ]
    if output_mode == "logits":
        if previous_target_logits_mask is None:
            raise RuntimeError("Logit-aware Correction requires previous feature masks")
        conditioning["previous_logits_mask"] = previous_target_logits_mask[
            :, position : position + 1
        ]
        if previous_rank_features is not None:
            conditioning["previous_rank_features"] = previous_rank_features[
                :, position : position + 1
            ]
        else:
            if previous_target_logits is None:
                raise RuntimeError("Logit-aware Correction requires previous logits")
            conditioning["previous_logits"] = previous_target_logits[
                :, position : position + 1
            ]
    return conditioning
