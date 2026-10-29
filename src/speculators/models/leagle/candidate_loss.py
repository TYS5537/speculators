"""Streaming forward-KL with one set of auxiliary gradients across TTT steps.

The backbone and dense teacher are detached. Compute the two linear gradients
while each vocabulary chunk is live, then let autograd deliver their accumulated
values to the parameters once. No chunk graph or dense logits survive for backward.
"""

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional


class _CandidateLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, down_weight, up_weight, down_grad, up_grad):
        ctx.save_for_backward(down_grad, up_grad)
        ctx.down_dtype, ctx.up_dtype = down_weight.dtype, up_weight.dtype
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        down, up = ctx.saved_tensors
        return (
            None,
            None if down is None else (down * grad_output).to(ctx.down_dtype),
            None if up is None else (up * grad_output).to(ctx.up_dtype),
            None,
            None,
        )


class CandidateLossAccumulator:
    """First-order KL gradients, summed across all tokens and TTT steps.

    Buffers occupy O(V * R + R * H), independent of the number of chunks/steps.
    Using a single final autograd node preserves normal DDP/FSDP gradient hooks
    and upstream loss scaling. This does not call backward inside model.forward.
    """

    def __init__(self, head):
        self.head = head
        self.value = head.down_proj.weight.new_zeros((), dtype=torch.float32)
        self.need_down = torch.is_grad_enabled() and head.down_proj.weight.requires_grad
        self.need_up = torch.is_grad_enabled() and head.up_proj.weight.requires_grad
        self.down_grad = None
        self.up_grad = None

    def _accumulate_gradients(
        self, hidden, reduced, coarse, log_q, target_p, mask, scale
    ):
        if not (self.need_down or self.need_up):
            return
        # d KL(p || q) / d logits = q - p. Cast at the same boundary as
        # log_softmax(..., dtype=float32)'s gradient into the linear output.
        grad = log_q.exp().sub_(target_p).mul_(mask[:, None]).mul_(scale)
        grad = grad.to(coarse.dtype)
        if self.need_up:
            update = grad.transpose(0, 1) @ reduced
            if self.up_grad is None:
                self.up_grad = update.float()
            else:
                self.up_grad.add_(update)
        if self.need_down:
            reduced_grad = grad @ self.head.up_proj.weight
            update = reduced_grad.transpose(0, 1) @ hidden
            if self.down_grad is None:
                self.down_grad = update.float()
            else:
                self.down_grad.add_(update)

    def _chunk(self, hidden, teacher, mask, top_k, scale):
        reduced = self.head.down_proj(hidden)
        coarse = self.head.up_proj(reduced)
        log_q = functional.log_softmax(coarse, dim=-1, dtype=torch.float32)
        log_p = functional.log_softmax(teacher, dim=-1, dtype=torch.float32)
        target_p = log_p.exp()
        loss = ((target_p * (log_p - log_q)).sum(-1) * mask).sum()
        candidates = coarse.topk(top_k, dim=-1, sorted=False).indices
        recalled = (candidates == teacher.argmax(-1, keepdim=True)).any(-1)
        recall = (recalled & mask).sum(dtype=torch.float32)
        self._accumulate_gradients(
            hidden, reduced, coarse, log_q, target_p, mask, scale
        )
        return loss, recall

    @torch.no_grad()
    def add(
        self,
        hidden_states,
        full_weight,
        mask,
        *,
        top_k,
        chunk_size,
        teacher_logits=None,
        normalize=False,
        loss_scale=1.0,
    ):
        """Add a step; return raw KL sum, top-1 recall count and valid count.

        Pass existing dense draft logits to avoid repeating W h. Their storage
        is read only during this call and is never saved for backward. Standalone
        callers may omit them to compute W h once per chunk.
        """
        vocab_size = self.head.up_proj.out_features
        if chunk_size < 1 or not 1 <= top_k <= vocab_size:
            raise ValueError("Invalid candidate loss chunk size or top_k")
        hidden = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
        valid = mask.reshape(-1).bool()
        if hidden.shape[0] != valid.shape[0] or hidden.shape[0] == 0:
            raise ValueError(
                "Candidate loss needs nonempty, aligned hidden states and mask"
            )
        if teacher_logits is not None:
            if teacher_logits.shape != (*hidden_states.shape[:-1], vocab_size):
                raise ValueError(
                    "Candidate teacher logits must align with hidden states"
                )
            teacher_logits = teacher_logits.detach().reshape(-1, vocab_size)
        count = valid.sum(dtype=torch.float32)
        scale = loss_scale / count.clamp_min(1) if normalize else loss_scale
        loss_sum = self.value.new_zeros(())
        recall_sum = loss_sum.clone()
        for start in range(0, hidden.shape[0], chunk_size):
            end = start + chunk_size
            teacher = (
                functional.linear(hidden[start:end], full_weight.detach())
                if teacher_logits is None
                else teacher_logits[start:end]
            )
            loss, recall = self._chunk(
                hidden[start:end], teacher, valid[start:end], top_k, scale
            )
            loss_sum.add_(loss)
            recall_sum.add_(recall)
        self.value.add_(loss_sum * scale)
        return loss_sum, recall_sum, count

    def loss(self):
        """Attach the accumulated first-order gradients once, after all updates."""
        return _CandidateLoss.apply(
            self.value,
            self.head.down_proj.weight,
            self.head.up_proj.weight,
            self.down_grad,
            self.up_grad,
        )
