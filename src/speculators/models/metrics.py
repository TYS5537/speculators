import json
import math
from collections.abc import Callable
from functools import cache

import torch
from torch.utils.checkpoint import checkpoint

# Compatibility re-exports: use the unchanged upstream eager primitives only.
# Keep local JS precision, stable NLA, dispatch and reduction policies below.
from speculators.losses.eager import (
    ce_loss,
    kl_div_loss,
    lk_hybrid_loss,
    reverse_kl_div_loss,
    tv_loss,
)
from speculators.losses.utils import (
    dflash_loss_decay,
    dpace_loss_decay,  # noqa: F401 -- Preserve the legacy public import path.
    exp_loss_decay,  # noqa: F401 -- Preserve the legacy public import path.
)

_EPS = 1e-5

LossConfig = dict[
    str, tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]
]


def compute_accuracy_single_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor | None,  # shape: [1, seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, seq_len]
):
    """Compute full and conditional accuracy counts for a single speculative step.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: If provided, restricts accuracy to masked positions.
        prev_correct: Boolean mask of positions correct so far. Updated in place
            via logical AND with the current step's correctness.

    Returns:
        Tuple of (full_correct, full_total, cond_correct, cond_total) as raw
        counts suitable for distributed reduction before computing ratios.
    """
    correct = pred_ids == target_ids
    cond_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)
    if prev_correct is not None:
        cond_total = prev_correct.sum().float()
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))

    correct_sum = correct.float().sum()
    full_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)

    return correct_sum, full_total, correct_sum.clone(), cond_total


@torch.no_grad()
def compute_accuracy_multi_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    num_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-position correct/total counts across multiple speculative steps.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: Boolean mask selecting positions to evaluate.
        pos_idx: Position index within each speculative block (e.g. 0,1,2,3,0,1,2,3).
        num_pos: Number of distinct positions (i.e. block size).

    Returns:
        Tuple of (correct_per_pos, total_per_pos) both with shape [num_pos].
        Overall counts can be derived by summing these.
    """
    correct = pred_ids == target_ids
    correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    pos_idx = torch.masked_select(pos_idx, loss_mask.to(torch.bool))

    correct_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    total_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    correct_per_pos.scatter_add_(0, pos_idx, correct.float())
    total_per_pos.scatter_add_(0, pos_idx, torch.ones_like(correct, dtype=torch.float))

    return correct_per_pos, total_per_pos  # shape: [num_pos], [num_pos]


def js_div_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position Jensen-Shannon divergence between draft and target.

    ``JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)`` with ``m = (p + q) / 2``.
    Symmetric and bounded by ``log 2`` (Lin 1991, "Divergence measures based on
    the Shannon entropy"), it balances forward KL's mass-covering pull with
    reverse KL's mode-seeking pull and keeps gradients finite where either
    distribution assigns near-zero probability. Compared to plain KL, this
    avoids unbounded penalties on tokens the target barely supports; compared
    to TV, it provides smoother, better-conditioned gradients for draft
    training.

    Args:
        logits: Draft model logits (log-softmax applied internally).
        targets: Target model logits (log-softmax applied internally).

    Returns:
        Per-position JS divergence with shape [1, seq_len].
    """
    draft_logq = torch.nn.functional.log_softmax(logits, dim=-1)
    target_logp = torch.nn.functional.log_softmax(targets, dim=-1)
    # log m = log((p + q) / 2), computed in log space for stability
    log_m = torch.logaddexp(draft_logq, target_logp) - math.log(2.0)
    kl_target_to_mix = torch.nn.functional.kl_div(
        log_m, target_logp, reduction="none", log_target=True
    ).sum(dim=-1)
    kl_draft_to_mix = torch.nn.functional.kl_div(
        log_m, draft_logq, reduction="none", log_target=True
    ).sum(dim=-1)
    elementwise_loss = 0.5 * (kl_target_to_mix + kl_draft_to_mix)  # [1, seq_len]

    return elementwise_loss  # noqa: RET504


def neg_log_acceptance_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position negative log-acceptance (LK) loss.

    The speculative-decoding acceptance rate equals the draft/target distribution
    overlap, ``alpha = sum_v min(p_v, q_v)`` (the same quantity computed in
    ``tv_loss``). This loss is ``-log(alpha)``. Its gradient is
    ``(1 / alpha) * grad(TV)``: the ``1 / alpha`` factor amplifies the otherwise
    vanishing TV gradient when overlap is low (early training), giving TV's
    acceptance-optimal target a usable gradient from a cold start. When the target
    is a point mass, this loss reduces to cross-entropy.

    Compute the overlap in log space: a probability floor would make the loss
    constant, with zero gradients, for low but representable acceptance rates.
    ``minimum`` splits its derivative equally at ties, just as the eager TV loss.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position negative log-acceptance with shape [1, seq_len].
    """
    draft_logp = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_logp = torch.nn.functional.log_softmax(targets, dim=-1, dtype=torch.float32)
    log_overlap_terms = torch.minimum(draft_logp, target_logp)
    # Center explicitly so logsumexp backward does not subtract two large
    # negative numbers (and lose gradient precision) for extremely low overlap.
    offset = log_overlap_terms.amax(dim=-1, keepdim=True).detach()
    elementwise_loss = -(
        offset.squeeze(-1) + torch.logsumexp(log_overlap_terms - offset, dim=-1)
    )

    return elementwise_loss  # noqa: RET504


def prefix_product_weights(
    scores: torch.Tensor,  # [1, T]
    block_size: int,
    start_pos: int = 0,
) -> torch.Tensor:
    """Per-position prefix-product weights within each block.

    Slot ``start_pos`` gets weight 1; later slots get the product of prior
    ``scores`` in the draft range. Slots before ``start_pos`` get 0.
    """
    num_blocks = scores.shape[1] // block_size
    blocks = scores.reshape(num_blocks, block_size)
    weights = torch.zeros_like(blocks)
    draft = blocks[:, start_pos:]
    pref = torch.ones_like(draft)
    if draft.shape[1] > 1:
        pref[:, 1:] = draft[:, :-1].cumprod(dim=-1)
    weights[:, start_pos:] = pref
    return weights.reshape_as(scores)


def position_weights(
    pos_idx: torch.Tensor,
    block_size: int,
    gamma: float,
    sample_from_anchor: bool = True,
    adaptive_scores: torch.Tensor | None = None,
    decay_mix: float = 0.0,
) -> torch.Tensor:
    """Fixed decay, adaptive prefix weights, or ``decay_mix`` convex mix."""
    decay = dflash_loss_decay(
        pos_idx, gamma=gamma, sample_from_anchor=sample_from_anchor
    )
    if adaptive_scores is None:
        return decay
    start_pos = 0 if sample_from_anchor else 1
    adaptive = prefix_product_weights(
        adaptive_scores, block_size=block_size, start_pos=start_pos
    )
    return decay_mix * decay + (1.0 - decay_mix) * adaptive


# ``tv`` uses Triton on CUDA/ROCm and eager operations on CPU/NPU. NLA instead
# uses stable log-space operations; CUDA/ROCm chunks and recomputes intermediate
# activations to bound memory without the unstable subtraction ``1 - TV``.
# Triton imports remain lazy, so this module also works without Triton installed.


@cache
def _fused_kernel(name: str):
    """Import and cache a fused kernel by name; ``None`` if Triton is unavailable."""
    try:
        from speculators.models import fused_tv_loss as mod  # noqa: PLC0415
    except ImportError:
        return None
    return getattr(mod, name)


def tv_loss_fused_or_eager(logits: torch.Tensor, targets: torch.Tensor):
    """TV loss: fused Triton on CUDA/ROCm (fp32), eager ``tv_loss`` on CPU/NPU."""
    if logits.is_cuda:
        kernel = _fused_kernel("fused_tv_loss")
        if kernel is not None:
            return kernel(logits, targets)
    return tv_loss(logits, targets)


def chunked_neg_log_acceptance_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    token_chunk_size: int = 128,
) -> torch.Tensor:
    """Stable NLA with token-bounded intermediates and backward recomputation.

    This is a PyTorch fallback, not a fused Triton kernel. Non-reentrant
    checkpointing retains input references instead of all FP32 log-probability
    activations. The intermediates occupy O(token_chunk_size * vocab) space;
    existing input/output tensors still remain resident. It costs extra dispatch
    and recomputation compared with fused TV, whose path is unchanged.
    """
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if logits.shape != targets.shape:
        raise ValueError("Draft and target logits must have the same shape")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape_as(flat_logits)
    if flat_logits.shape[0] == 0:
        return neg_log_acceptance_loss(logits, targets)
    recompute = torch.is_grad_enabled() and (
        logits.requires_grad or targets.requires_grad
    )
    chunks = []
    for start in range(0, flat_logits.shape[0], token_chunk_size):
        draft_chunk = flat_logits[start : start + token_chunk_size]
        target_chunk = flat_targets[start : start + token_chunk_size]
        if recompute:
            loss_chunk = checkpoint(
                neg_log_acceptance_loss,
                draft_chunk,
                target_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            loss_chunk = neg_log_acceptance_loss(draft_chunk, target_chunk)
        chunks.append(loss_chunk)
    return torch.cat(chunks).reshape(logits.shape[:-1])


def nla_loss_fused_or_eager(logits: torch.Tensor, targets: torch.Tensor):
    """NLA: chunked/recomputed PyTorch on CUDA/ROCm, eager log-space on CPU/NPU.

    The historical dispatcher name is retained for loss-config compatibility.
    CUDA NLA no longer uses a Triton kernel; see the chunked helper's cost note.
    """
    if logits.is_cuda:
        return chunked_neg_log_acceptance_loss(logits, targets)
    return neg_log_acceptance_loss(logits, targets)


_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": kl_div_loss,
    "rkl": reverse_kl_div_loss,
    "jsd": js_div_loss,
    "ce": ce_loss,
    "tv": tv_loss_fused_or_eager,
    "nla": nla_loss_fused_or_eager,
    "lk_hybrid": lk_hybrid_loss,
}


def resolve_loss_config(spec: str) -> LossConfig:
    """Parse a loss spec into ``{name: (loss_fn, weight)}``.

    Accepts either a plain loss name (``"kl_div"``) or a JSON dict mapping
    loss names to weights (``'{"ce": 0.1, "tv": 0.9}'``).
    """
    if spec in _LOSS_FN_MAP:
        return {spec: (_LOSS_FN_MAP[spec], 1.0)}

    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError:
        raise ValueError(
            f"Unknown loss function '{spec}'. Pass a known name "
            f"({sorted(_LOSS_FN_MAP.keys())}) or a JSON dict, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'.'
        ) from None

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            "Loss config must be a non-empty JSON dict mapping loss names to weights, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'. Got: {spec}'
        )

    config: LossConfig = {}
    for name, weight in parsed.items():
        if name not in _LOSS_FN_MAP:
            raise ValueError(
                f"Unknown loss function '{name}' in loss config. "
                f"Choose from: {sorted(_LOSS_FN_MAP.keys())}"
            )
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Loss weight for '{name}' must be a number, "
                f"got {type(weight).__name__}"
            )
        config[name] = (_LOSS_FN_MAP[name], float(weight))

    return config


def compound_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[..., torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a weighted sum of loss terms.

    Each entry in *loss_config* maps a name to ``(loss_fn, weight)``; the
    result is ``sum(weight * loss_function(logits, targets, ..., loss_fn))``
    over all entries.

    Returns the total loss and a dict of per-term (unweighted) scalar losses
    keyed as ``"{name}_loss"``.  When the config contains a single term the
    dict is empty (the overall loss already captures it).
    """
    total = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    term_losses: dict[str, torch.Tensor] = {}
    multi = len(loss_config) > 1
    for name, (fn, weight) in loss_config.items():
        term = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=fn,
            decay_fn=decay_fn,
        )
        if multi:
            term_losses[f"{name}_loss"] = term.detach()
        total = total + weight * term
    return total, term_losses


def loss_function(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[..., torch.Tensor] | None = None,
):
    """Compute masked, optionally position-decayed training loss.

    Args:
        logits: Draft model logits.
        targets: Target model logits.
        loss_mask: Boolean mask selecting positions to include in the loss.
        pos_idx: Position indices within each speculative block.
        loss_fn: Per-position loss function (default: kl_div_loss).
        decay_fn: Optional position-dependent decay weighting function.

    Returns:
        Scalar mean loss across the batch.
    """
    elementwise_loss = loss_fn(logits, targets)  # shape: [1, seq_len]

    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if decay_fn is not None:
        decay_mult = decay_fn(
            pos_idx.to(elementwise_loss.dtype), elementwise_loss=elementwise_loss
        )
        elementwise_loss = elementwise_loss * decay_mult

    denominator = loss_mask.sum(dim=1) + _EPS

    batch_loss = torch.sum(elementwise_loss, dim=1) / denominator  # shape: [1]
    return batch_loss.mean()  # shape: []


def compute_accepted_length_counts(
    correct: torch.Tensor,  # shape: [num_blocks, num_draft_slots]
    valid: torch.Tensor,  # shape: [num_blocks, num_draft_slots]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count accepted length per speculative block as raw sum/total counts.

    A block is accepted up to its first wrong draft slot, so its length is the
    leading run of correct slots plus the verifier's always-emitted bonus token
    (vLLM's convention). Forming the run inside the block preserves the
    correlation between slots; a product of per-slot marginals discards it and
    understates the result.

    Args:
        correct: Whether each draft slot matched the target.
        valid: Whether each draft slot is trained on (block-shaped loss mask).

    Returns:
        Tuple of (accepted_length_sum, valid_block_total) as raw counts suitable
        for distributed reduction before computing the ratio.
    """
    accepted = torch.logical_and(correct, valid).to(torch.float32)
    per_block_len = accepted.cumprod(dim=-1).sum(dim=-1) + 1.0
    block_valid = valid.any(dim=-1).to(torch.float32)
    return (per_block_len * block_valid).sum(), block_valid.sum()


def resolve_training_loss(loss_fn: str, **kwargs) -> LossConfig:
    """Keep legacy numerical implementations unless the upstream recipe is selected."""
    implementation = kwargs.get("loss_implementation")
    if implementation is None:
        implementation = (
            "fused" if kwargs.get("training_recipe") == "upstream" else "legacy"
        )
    if implementation == "legacy":
        return resolve_loss_config(loss_fn)
    from speculators.losses import (  # noqa: PLC0415
        resolve_loss_config as resolve_upstream,
    )

    return resolve_upstream(loss_fn, implementation)
