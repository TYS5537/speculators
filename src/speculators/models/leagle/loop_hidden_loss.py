"""Intermediate-loop supervision in the frozen teacher's pre-LM space."""

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from speculators.losses.utils import exp_loss_decay, reduce_masked_loss


class LoopHiddenAlignment:
    """Mean normalized-hidden MSE across intermediate loops, summed over TTT.

    The caller branches after intermediate stack/sublayer visits, excluding the
    final visit of each repeated unit. Sublayer modes average across layers too.
    Only the branch is normalized; the recurrent hidden and KV are never changed
    or detached. Teacher states already have the same normalization used by the
    main output loss, and are never projected through the vocabulary head here.
    """

    def __init__(self, model, teacher, docs, valid, decay):
        if teacher.shape[:2] != docs.shape or teacher.shape[-1] != model.hidden_size:
            raise ValueError(
                "Loop HS teacher states must match sequence and hidden size"
            )
        self.norm = model.norm
        self.teacher = teacher.detach()
        self.docs, self.valid = docs, valid
        self.decay = decay
        self.loop_count = model.config.loop_supervision_count
        self.checkpoint = model.activation_checkpointing
        self.loss = teacher.new_zeros((), dtype=torch.float32)

    def _loss(self, hidden, teacher, mask, step):
        predicted = self.norm(hidden)
        # Preserve native RMSNorm precision, then compute the comparison and
        # reduction in FP32, including under BF16 autocast.
        error = (predicted.float() - teacher.float()).square().mean(dim=-1)
        positions = torch.full_like(mask, step, dtype=torch.long)
        return reduce_masked_loss(
            error,
            mask,
            positions,
            partial(exp_loss_decay, gamma=self.decay),
        )

    def add(self, hidden, step):
        width = hidden.shape[1] - step
        if width <= 0:
            return
        # Identical alignment to the main TTT objective: shift teacher/mask,
        # retain the draft start, and exclude cross-document/padding positions.
        mask = self.valid[:, step:] & (self.docs[:, step:] == self.docs[:, :width])
        arguments = (hidden[:, :width], self.teacher[:, step:], mask, step)
        value = (
            checkpoint(
                self._loss,
                *arguments,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            if self.checkpoint and torch.is_grad_enabled()
            else self._loss(*arguments)
        )
        self.loss = self.loss + value / self.loop_count
