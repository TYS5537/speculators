"""Budgeted, forward-local teacher probabilities for eager CE/TV/forward KL."""

import torch
from torch.nn import functional

from speculators.losses import eager
from speculators.models.leagle.shared_loss import normalized_distribution
from speculators.models.metrics import tv_loss_fused_or_eager


def validate_teacher_cache(loss_config, device):
    known = {
        "ce": (eager.ce_loss,),
        "tv": (eager.tv_loss, tv_loss_fused_or_eager),
        "kl_div": (eager.kl_div_loss,),
    }
    if not loss_config or any(
        function not in known.get(name, ())
        or (device.type == "cuda" and function is tv_loss_fused_or_eager)
        for name, (function, _) in loss_config.items()
    ):
        raise ValueError(
            "LEAGLE teacher cache requires eager CE, TV or forward KL; "
            "use --loss-implementation eager (legacy also works on CPU/NPU)"
        )


@torch.no_grad()
def _distribution(hidden, weight, *, shared_normalization=False):
    logits = functional.linear(hidden, weight)
    probabilities = (
        normalized_distribution(logits)[1]
        if shared_normalization
        else logits.softmax(-1, dtype=torch.float32)
    )
    return probabilities, logits.argmax(-1)


class TeacherCache:
    """Cache a fixed prefix or supervised subset, recomputing uncached positions.

    A prefix avoids LRU thrashing as every TTT step revisits almost all tokens.
    The budget covers persistent probabilities, IDs and optional position mapping,
    not temporary chunk workspace.
    This object is created inside each forward and never outlives its loss graph.
    """

    @torch.no_grad()
    def __init__(
        self,
        hidden,
        weight,
        *,
        budget_bytes,
        chunk_size,
        priority_mask=None,
        shared_normalization=False,
    ):
        self.shared_normalization = shared_normalization
        self.slots = None
        positions = None
        if priority_mask is not None:
            if priority_mask.shape != hidden.shape[:2] or hidden.shape[0] != 1:
                raise ValueError("Teacher cache mask must align with one sequence")
            metadata_bytes = hidden.shape[1] * 8
            if budget_bytes <= metadata_bytes:
                raise ValueError("Teacher cache budget cannot fit position metadata")
            budget_bytes -= metadata_bytes
            self.slots = torch.full(
                (hidden.shape[1],), -1, device=hidden.device, dtype=torch.long
            )
            positions = priority_mask[0].bool().nonzero(as_tuple=True)[0]
        bytes_per_row = 4 * weight.shape[0] + 8
        available = hidden.shape[1] if positions is None else positions.numel()
        self.rows = min(available, budget_bytes // bytes_per_row)
        if (self.rows < 1 and available) or chunk_size < 1:
            raise ValueError("Teacher cache budget must fit at least one position")
        if positions is not None:
            positions = positions[: self.rows]
            self.slots[positions] = torch.arange(self.rows, device=hidden.device)
        self.probs = hidden.new_empty(
            (1, self.rows, weight.shape[0]), dtype=torch.float32
        )
        self.ids = torch.empty((1, self.rows), device=hidden.device, dtype=torch.long)
        for start in range(0, self.rows, chunk_size):
            end = min(start + chunk_size, self.rows)
            selected = slice(start, end) if positions is None else positions[start:end]
            probabilities, ids = _distribution(
                hidden[:, selected], weight, shared_normalization=shared_normalization
            )
            self.probs[:, start:end].copy_(probabilities)
            self.ids[:, start:end].copy_(ids)

    @torch.no_grad()
    def get(self, hidden, weight, *, start=0, indices=None):
        if self.slots is not None:
            positions = (
                torch.arange(start, start + hidden.shape[1], device=hidden.device)
                if indices is None
                else indices
            )
            return self._indexed_get(hidden, weight, positions)
        # Selected indices are ascending, as produced by mask.nonzero().
        if indices is None:
            count = max(0, min(hidden.shape[1], self.rows - start))
            probabilities = self.probs[:, start : start + count]
            ids = self.ids[:, start : start + count]
        else:
            cached = indices[indices < self.rows]
            count = cached.numel()
            probabilities, ids = self.probs[:, cached], self.ids[:, cached]
        if count == hidden.shape[1]:
            return probabilities, ids
        extra_probs, extra_ids = _distribution(
            hidden[:, count:], weight, shared_normalization=self.shared_normalization
        )
        if count == 0:
            return extra_probs, extra_ids
        return (
            torch.cat((probabilities, extra_probs), dim=1),
            torch.cat((ids, extra_ids), dim=1),
        )

    def _indexed_get(self, hidden, weight, positions):
        slots = self.slots[positions]
        hits = (slots >= 0).nonzero(as_tuple=True)[0]
        misses = (slots < 0).nonzero(as_tuple=True)[0]
        if not misses.numel():
            return self.probs[:, slots], self.ids[:, slots]
        extra_probs, extra_ids = _distribution(
            hidden[:, misses], weight, shared_normalization=self.shared_normalization
        )
        if not hits.numel():
            return extra_probs, extra_ids
        probabilities = extra_probs.new_empty(
            (1, positions.numel(), self.probs.shape[-1])
        )
        ids = self.ids.new_empty((1, positions.numel()))
        probabilities[:, hits], ids[:, hits] = (
            self.probs[:, slots[hits]],
            self.ids[:, slots[hits]],
        )
        probabilities[:, misses], ids[:, misses] = extra_probs, extra_ids
        return probabilities, ids


def cached_terms(logits, probabilities, ids, loss_config, metrics):
    terms = []
    overlap = None
    for name in loss_config:
        if name == "ce":
            value = functional.cross_entropy(
                logits.flatten(0, 1), ids.flatten(), reduction="none", ignore_index=-100
            ).reshape_as(ids)
        elif name == "tv":
            value = 1.0 - torch.minimum(
                logits.softmax(-1, dtype=torch.float32), probabilities
            ).sum(-1)
            if metrics:
                overlap = (1.0 - value.detach()).clamp(0, 1)
        else:  # Validated forward KL.
            value = functional.kl_div(
                logits.log_softmax(-1, dtype=torch.float32),
                probabilities,
                reduction="none",
                log_target=False,
            ).sum(-1)
        terms.append(value)
    if metrics and overlap is None:
        with torch.no_grad():
            overlap = (
                torch.minimum(logits.softmax(-1, dtype=torch.float32), probabilities)
                .sum(-1)
                .clamp(0, 1)
            )
    return tuple(terms), overlap
