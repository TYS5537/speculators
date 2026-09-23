"""Content-conditioned causal convolutions used by the MMuse backbone."""

import torch
from torch import nn


def grouped_dynamic_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    *,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    """Apply a causal grouped convolution without crossing draft blocks."""
    if hidden_states.shape[-1] != num_groups * group_size:
        raise ValueError("Grouped-conv hidden size does not match its groups")
    if hidden_states.shape[-2] % block_size != 0:
        raise ValueError("Grouped-conv sequence length must be divisible by block_size")
    expected_delta_shape = (*hidden_states.shape[:-1], taps, num_groups)
    if delta.shape != expected_delta_shape:
        raise ValueError(
            f"Expected dynamic coefficients {expected_delta_shape}, "
            f"got {tuple(delta.shape)}"
        )

    blocks = hidden_states.reshape(-1, block_size, num_groups, group_size)
    dynamic = delta.reshape(-1, block_size, taps, num_groups)
    coefficients = base.reshape(1, 1, taps, num_groups, group_size) + (
        dynamic.unsqueeze(-1)
    )
    output = coefficients[:, :, 0] * blocks
    for tap in range(1, taps):
        if tap >= block_size:
            continue
        shifted = torch.cat(
            [
                torch.zeros_like(blocks[:, :tap]),
                coefficients[:, tap:, tap] * blocks[:, :-tap],
            ],
            dim=1,
        )
        output = output + shifted
    return output.reshape_as(hidden_states)


class DFlash2GroupedConv(nn.Module):
    """DFlash2 content-conditioned causal convolution around one sublayer."""

    def __init__(
        self,
        hidden_size: int,
        *,
        taps: int,
        group_size: int,
        block_size: int,
    ) -> None:
        super().__init__()
        if taps <= 0:
            raise ValueError(f"conv_kernel_size must be > 0, got {taps}")
        if group_size <= 0 or hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}"
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(torch.empty(2, taps, hidden_size))
        self.kernel_projection = nn.Linear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
        )

    def reset_identity(self) -> None:
        """Start as an exact identity while leaving both paths trainable."""
        with torch.no_grad():
            self.base_kernel.zero_()
            self.base_kernel[:, 0].fill_(1.0)
            self.kernel_projection.weight.zero_()

    def _convolve(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        side: int,
    ) -> torch.Tensor:
        return grouped_dynamic_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            block_size=self.block_size,
            num_groups=self.num_groups,
            group_size=self.group_size,
            taps=self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1],
            2,
            self.taps,
            self.num_groups,
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], 0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self,
        hidden_states: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, 1)
