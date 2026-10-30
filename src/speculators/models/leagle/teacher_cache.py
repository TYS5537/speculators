"""Budgeted, forward-local teacher probabilities for eager CE/TV/forward KL."""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class TeacherLookupPlan:
    width: int
    slots: torch.Tensor
    hits: torch.Tensor | None = None
    misses: torch.Tensor | None = None


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
        cache_indices=False,
    ):
        self.shared_normalization = shared_normalization
        self.cache_indices = cache_indices
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
    def plan_chunks(self, length, chunk_size, *, start=0, indices=None):
        """Build immutable lookup indices once per TTT step, outside checkpoint.

        One small device-to-host read replaces per-chunk dynamic nonzero calls.
        Pack all device indices into one transfer and retain only slice views for
        backward replay. Prefix queries without compaction already use arithmetic.
        Plans belong to this forward only; their small metadata is extra workspace
        outside the probability cache budget.
        """
        if not self.cache_indices or (self.slots is None and indices is None):
            return None
        if chunk_size < 1 or length < 0:
            raise ValueError("Teacher lookup plans need valid length and chunk size")
        values = (
            self.slots[start : start + length]
            if indices is None
            else self.slots[indices]
            if self.slots is not None
            else indices
        ).tolist()
        if len(values) != length:
            raise ValueError("Teacher lookup indices must align with the query length")
        packed, layouts = [], []
        indexed = self.slots is not None
        for offset in range(0, length, chunk_size):
            part = values[offset : offset + chunk_size]
            hits = [i for i, slot in enumerate(part) if slot >= 0] if indexed else []
            misses = [i for i, slot in enumerate(part) if slot < 0] if indexed else []
            slots = (
                [part[i] for i in hits]
                if indexed
                else [i for i in part if i < self.rows]
            )
            layouts.append((len(part), len(packed), len(slots), len(misses)))
            packed.extend(slots)
            packed.extend(hits)
            packed.extend(misses)
        metadata = torch.tensor(packed, device=self.probs.device, dtype=torch.long)
        return tuple(
            TeacherLookupPlan(
                width,
                metadata[offset : offset + count],
                metadata[offset + count : offset + 2 * count] if indexed else None,
                metadata[offset + 2 * count : offset + 2 * count + missing]
                if indexed
                else None,
            )
            for width, offset, count, missing in layouts
        )

    @torch.no_grad()
    def get(self, hidden, weight, *, start=0, indices=None, plan=None):
        if plan is not None:
            if plan.width != hidden.shape[1]:
                raise ValueError("Teacher lookup plan must align with the query length")
            if plan.hits is not None:
                return self._indexed_get(hidden, weight, plan=plan)
            return self._complete_prefix(
                hidden, weight, self.probs[:, plan.slots], self.ids[:, plan.slots]
            )
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
        return self._complete_prefix(hidden, weight, probabilities, ids)

    def _complete_prefix(self, hidden, weight, probabilities, ids):
        count = probabilities.shape[1]
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

    def _indexed_get(self, hidden, weight, positions=None, *, plan=None):
        if plan is None:
            slots = self.slots[positions]
            hits = (slots >= 0).nonzero(as_tuple=True)[0]
            misses = (slots < 0).nonzero(as_tuple=True)[0]
            cached = slots[hits] if misses.numel() else slots
        else:
            cached, hits, misses = plan.slots, plan.hits, plan.misses
        if not misses.numel():
            return self.probs[:, cached], self.ids[:, cached]
        extra_probs, extra_ids = _distribution(
            hidden[:, misses], weight, shared_normalization=self.shared_normalization
        )
        if not hits.numel():
            return extra_probs, extra_ids
        probabilities = extra_probs.new_empty(
            (1, hidden.shape[1], self.probs.shape[-1])
        )
        ids = self.ids.new_empty((1, hidden.shape[1]))
        probabilities[:, hits], ids[:, hits] = (
            self.probs[:, cached],
            self.ids[:, cached],
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
