"""Named tensor boundaries for MMuse training, Selector and Correction rollout.

These tuple-compatible containers only hold references: they perform no tensor
operations, validation or state registration. Field order preserves the previous
positional returns. B is the number of blocks, T the block size, H the backbone
width, C the Correction width, R its compact rank and V the draft vocabulary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from torch import Tensor

    from speculators.models.mmuse.correction import CorrectionCache


class SelectorConditioning(NamedTuple):
    """Teacher-forced inputs for joint Selector/Correction training.

    Previous IDs use the verifier vocabulary [B, T]. Current embeddings [B, T, H]
    are optional and frozen by the producer, while compact features [B, T, R]
    retain the producer's gradient path. Their validity mask is [B, T]. A disabled
    Selector keeps the original previous IDs and leaves every other field None.
    """

    selector_loss: Tensor | None
    previous_token_ids: Tensor
    current_token_embeddings: Tensor | None
    previous_rank_features: Tensor | None
    previous_logits_mask: Tensor | None


class SelectorCorrectionInputs(NamedTuple):
    """A static Selector path and its predecessor-distribution features.

    Both ID tensors are verifier-vocabulary IDs [B, T], not draft IDs. Optional
    rank features [B, T, R] and their mask [B, T] are supplied together in logits
    mode. The existing producer owns token shifting and gradient boundaries.
    """

    current_token_ids: Tensor
    previous_token_ids: Tensor
    previous_rank_features: Tensor | None
    previous_logits_mask: Tensor | None


class SelectorPreviousCandidates(NamedTuple):
    """Shifted sparse proposal rows before compact Correction encoding.

    Candidate IDs and logits [B, T, K] remain in the draft vocabulary, unlike the
    verifier IDs in SelectorCorrectionInputs. The [B, T] mask marks valid previous
    rows; initial dense verifier logits are encoded and inserted separately.
    """

    candidate_ids: Tensor
    candidate_logits: Tensor
    mask: Tensor


class TrainingBlocks(NamedTuple):
    """Teacher-forced block inputs; both ID tensors use the verifier vocabulary.

    IDs and positions are [B, T], hidden is [B, T, H], optional base logits are
    [B, T, V]. The producer keeps the original indexing, shifts and tensor views.
    """

    token_ids: Tensor
    previous_token_ids: Tensor
    hidden: Tensor
    positions: Tensor
    base_logits: Tensor | None


class TeacherForcedCorrectionOutput(NamedTuple):
    """Flattened logits [1, B*T, V], states [B, T, C] and optional hidden [B, T, H].

    Returned before collaboration, diagnostics and losses. The existing forward
    guard remains responsible for rejecting missing logits from an invalid mode.
    """

    logits: Tensor | None
    causal_states: Tensor
    corrected_hidden: Tensor | None


class InitialLogitFeedback(NamedTuple):
    """Initial dense or online-Selector feedback, before the first rollout slot.

    Dense logits have shape [B, 1, V]; online rank features have shape [B, 1, R].
    Each active representation has a [B, 1] validity mask. All fields are None
    when logit feedback is unused or static conditioning already supplies it.
    """

    dense_logits: Tensor | None
    dense_mask: Tensor | None
    online_rank_features: Tensor | None
    online_mask: Tensor | None


class CorrectionStepOutput(NamedTuple):
    """One slot's logits [B, V], states [B, C], hidden [B, H], and next cache.

    The cache is the producer's original object, not a copy. Markov collaboration,
    sampling and recurrent feedback updates remain the caller's responsibility.
    """

    logits: Tensor
    causal_states: Tensor
    corrected_hidden: Tensor
    cache: CorrectionCache | None


class CorrectionRolloutOutput(NamedTuple):
    """All block slots, including the anchor slot when it is not sampled.

    Token IDs [B, T] use the draft vocabulary. Logits, Correction states and
    corrected hidden states have shapes [B, T, V], [B, T, C] and [B, T, H].
    The public rollout API still exposes only the plain (token_ids, logits) pair.
    """

    token_ids: Tensor
    logits: Tensor
    causal_states: Tensor
    corrected_hidden: Tensor
