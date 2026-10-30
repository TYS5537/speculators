"""Opt-in candidate KL with loss scaling applied during chunked backward.

Each autograd node represents a complete TTT step. Vocabulary intermediates
are recomputed, and weight products accumulate in FP32 before one cast per step,
matching the rounding boundary of an unchunked linear backward. No [L,V] tensor
is retained. This trades extra projections for closer reference gradients.
"""

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional


def _project(hidden, full_weight, down, up, collect_recall=False):
    teacher = functional.linear(hidden, full_weight)
    log_p = functional.log_softmax(teacher, dim=-1, dtype=torch.float32)
    reduced = functional.linear(hidden, down)
    return (
        reduced,
        functional.linear(reduced, up),
        log_p,
        teacher.argmax(-1, keepdim=True) if collect_recall else None,
    )


def _kl(coarse, log_p, mask):
    log_q = functional.log_softmax(coarse, dim=-1, dtype=torch.float32)
    return ((log_p.exp() * (log_p - log_q)).sum(-1) * mask).sum()


class _CandidateStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, full_weight, down, up, mask, scale, chunk_size, top_k):
        ctx.save_for_backward(hidden, full_weight, down, up, mask, scale)
        ctx.chunk_size = chunk_size
        ctx.autocast_enabled = torch.is_autocast_enabled(hidden.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(hidden.device.type)
        total = hidden.new_zeros((), dtype=torch.float32)
        recall = total.clone()
        for start in range(0, hidden.shape[0], chunk_size):
            end = start + chunk_size
            _, coarse, log_p, labels = _project(
                hidden[start:end], full_weight, down, up, bool(top_k)
            )
            total.add_(_kl(coarse, log_p, mask[start:end]))
            if top_k:
                ids = coarse.topk(top_k, dim=-1, sorted=False).indices
                hit = (ids == labels).any(-1)
                recall.add_((hit & mask[start:end]).sum(dtype=torch.float32))
        ctx.mark_non_differentiable(total, recall)
        return total * scale, total, recall

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, _grad_total, _grad_recall):
        hidden, full_weight, down, up, mask, scale = ctx.saved_tensors
        need_down, need_up = ctx.needs_input_grad[2:4]
        down_grad = torch.zeros_like(down, dtype=torch.float32) if need_down else None
        up_grad = torch.zeros_like(up, dtype=torch.float32) if need_up else None
        for start in range(0, hidden.shape[0], ctx.chunk_size):
            end = start + ctx.chunk_size
            part = hidden[start:end]
            with torch.autocast(
                hidden.device.type,
                enabled=ctx.autocast_enabled,
                dtype=ctx.autocast_dtype,
            ):
                reduced, coarse, log_p, _ = _project(part, full_weight, down, up)
            # Native log-softmax backward preserves its reduction and cast, with
            # the actual upstream scale already included. q-p alone is not the
            # same floating-point computation, even when the formulas agree.
            with torch.enable_grad(), torch.autocast(hidden.device.type, enabled=False):
                coarse = coarse.detach().requires_grad_(True)
                loss = _kl(coarse, log_p, mask[start:end]) * scale
                grad = torch.autograd.grad(loss, coarse, grad_outputs=grad_output)[0]
            with torch.autocast(hidden.device.type, enabled=False):
                if need_up:
                    up_grad.addmm_(grad.T.float(), reduced.float())
                if need_down:
                    # Retain the native low-precision activation-adjoint boundary.
                    reduced_grad = grad @ up
                    down_grad.addmm_(
                        reduced_grad.T.float(), part.to(reduced.dtype).float()
                    )
        return (
            None,
            None,
            down_grad.to(down.dtype) if need_down else None,
            up_grad.to(up.dtype) if need_up else None,
            None,
            None,
            None,
            None,
        )


class RecomputedCandidateLoss:
    """Keep complete-step rounding independent of main-head chunk boundaries.

    Detached hidden states and existing weights survive for backward; teacher
    and candidate vocabularies are only temporary chunks. AMP casts of candidate
    weights are shared across steps, as in a single native autocast forward.
    """

    def __init__(self, head):
        self.head = head
        self.weights = None
        self.value = head.down_proj.weight.new_zeros((), dtype=torch.float32)

    def add(
        self,
        hidden_states,
        full_weight,
        mask,
        *,
        top_k,
        chunk_size,
        normalize=False,
        loss_scale=1.0,
        collect_recall=True,
    ):
        vocab_size = self.head.up_proj.out_features
        if chunk_size < 1 or not 1 <= top_k <= vocab_size:
            raise ValueError("Invalid candidate loss chunk size or top_k")
        hidden = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
        valid = mask.reshape(-1).bool()
        if hidden.shape[0] != valid.shape[0] or hidden.shape[0] == 0:
            raise ValueError(
                "Candidate loss needs nonempty, aligned hidden states and mask"
            )
        if self.weights is None:
            self.weights = (
                self.head.down_proj.weight,
                self.head.up_proj.weight,
            )
            if torch.is_autocast_enabled(hidden.device.type):
                dtype = torch.get_autocast_dtype(hidden.device.type)
                self.weights = tuple(weight.to(dtype) for weight in self.weights)
        count = valid.sum(dtype=torch.float32)
        scale = torch.as_tensor(loss_scale, device=hidden.device, dtype=torch.float32)
        if normalize:
            scale = scale / count.clamp_min(1)
        value, raw, recall = _CandidateStep.apply(
            hidden,
            full_weight.detach(),
            *self.weights,
            valid,
            scale,
            chunk_size,
            top_k if collect_recall else 0,
        )
        self.value = self.value + value
        return raw, recall, count

    def loss(self):
        return self.value
