"""Read-only loss observations for LEAGLE's training metrics."""

from speculators.losses import tv_loss as fused_tv_loss
from speculators.losses.eager import tv_loss
from speculators.models.metrics import tv_loss_fused_or_eager


class TVOverlapProbe:
    """Observe an existing TV result without changing its value or autograd graph.

    Only known TV implementations have the required ``TV = 1 - overlap``
    contract. Custom losses and objectives without TV keep the metric fallback.
    Retain only the detached per-position vector, never vocabulary activations.
    """

    def __init__(self, loss_config):
        self.loss_config = loss_config
        self.overlap = None
        if loss_config is not None and "tv" in loss_config:
            function, weight = loss_config["tv"]
            if function in (tv_loss, tv_loss_fused_or_eager, fused_tv_loss):
                self.function = function
                self.loss_config = dict(loss_config)
                self.loss_config["tv"] = (self._observe, weight)

    def _observe(self, logits, targets):
        value = self.function(logits, targets)
        self.overlap = (1.0 - value.detach()).clamp(0, 1)
        return value
