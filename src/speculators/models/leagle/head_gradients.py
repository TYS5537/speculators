"""First-order FP32 frozen-head gradients without replaying vocabulary projections.

Only detached chunk-local inputs enter the temporary autograd graph. Parameter
hooks and the decoder graph run during the ordinary outer backward, where the
saved hidden gradient also receives distributed supervision/outer loss scaling.
"""

from functools import partial

import torch
from torch.autograd.function import once_differentiable

from speculators.losses.utils import exp_loss_decay, reduce_masked_loss


def supports_fp32_head(hidden):
    # BF16/FP16 round intermediate adjoints before reaching the hidden state.
    # Moving an arbitrary outer scale past those casts changes the gradients.
    return hidden.dtype == torch.float32 and not (
        torch.is_autocast_enabled(hidden.device.type)
        and torch.get_autocast_dtype(hidden.device.type) != torch.float32
    )


class _FrozenHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _hidden, value, gradient):
        ctx.save_for_backward(gradient)
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, scale):
        (gradient,) = ctx.saved_tensors
        return (gradient * scale).to(gradient.dtype), None, None


class FrozenHeadGradients:
    """Retain O(sequence * hidden) instead of checkpointing the frozen head.

    Use the original loss operations, including TV's tie subgradient and native CE
    precision. Local autograd never touches trainable parameters or calls backward
    on the model. This is intentionally a first-order optimization.
    """

    def __init__(self, hidden, loss_config, mask, step, decay):
        if not supports_fp32_head(hidden):
            raise ValueError(
                "Streaming main loss requires FP32 without low-precision autocast"
            )
        self.hidden, self.loss_config = hidden, loss_config
        self.mask, self.step, self.decay = mask, step, decay
        self.gradient = torch.zeros_like(hidden)
        self.coefficients = None

    def _coefficients(self, terms):
        # Different terms may have different dtypes. Derive their adjoints from
        # the exact full-step reduction, not an approximate per-chunk mean.
        positions = torch.full_like(self.mask, self.step, dtype=torch.long)
        coefficients = []
        for term, (_, weight) in zip(terms, self.loss_config.values(), strict=True):
            vector = torch.zeros_like(self.mask, dtype=term.dtype, requires_grad=True)
            value = weight * reduce_masked_loss(
                vector,
                self.mask,
                positions,
                partial(exp_loss_decay, gamma=self.decay),
            )
            coefficients.append(torch.autograd.grad(value, vector)[0])
        return coefficients

    def project(self, function, arguments, selected):
        if not supports_fp32_head(arguments[0]):
            raise ValueError(
                "Streaming main loss requires FP32 without low-precision autocast"
            )
        if arguments[2].requires_grad or arguments[3].requires_grad:
            raise ValueError("Streaming main loss requires frozen output matrices")
        with torch.enable_grad():
            local = arguments[0].detach().requires_grad_(True)
            result = function(local, *arguments[1:])
            terms = result[0]
            if self.coefficients is None:
                self.coefficients = self._coefficients(terms)
            coefficients = tuple(value[:, selected] for value in self.coefficients)
            # Ordinary training runs backward outside autocast. Preserve that
            # policy for this isolated head graph as well.
            with torch.autocast(device_type=local.device.type, enabled=False):
                (gradient,) = torch.autograd.grad(terms, local, coefficients)
        self.gradient[:, selected] = gradient
        return (tuple(term.detach() for term in terms), *result[1:])

    def attach(self, value):
        return _FrozenHeadLoss.apply(self.hidden, value.detach(), self.gradient)
