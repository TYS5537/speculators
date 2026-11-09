"""Second-stage retrieval training over a frozen LEAGLE draft."""

import torch

from speculators.models.leagle.candidate_head import LowRankCandidateHead
from speculators.models.leagle.chunked_loss import predict_chunks
from speculators.models.leagle.objective import select_candidate_positions
from speculators.models.leagle.svd_head import svd_factors


def prepare_candidate_training(model, *, initialization="svd"):
    """Initialize a missing head, then freeze everything except its two factors.

    The caller has explicitly loaded and validated the source checkpoint before
    updating its candidate configuration. Existing candidate weights are retained.
    This training policy is runtime-only; the saved model is an ordinary LEAGLE.
    """
    if initialization not in {"svd", "random"}:
        raise ValueError("Candidate initialization must be svd or random")
    if not model.config.candidate_head or not model.config.candidate_rerank:
        raise ValueError("Candidate-only training requires an auxiliary candidate head")
    if model.candidate_head is None:
        weight = model.lm_head.weight
        if weight.device.type != "cpu":
            raise ValueError(
                "Initialize candidate training before moving/sharding the draft"
            )
        with torch.random.fork_rng(devices=[]):
            head = LowRankCandidateHead(
                model.hidden_size,
                model.draft_vocab_size,
                model.config.candidate_head_rank,
                model.config.transformer_layer_config.initializer_range,
            ).to(dtype=weight.dtype)
        if initialization == "svd":
            up, down = svd_factors(weight, model.config.candidate_head_rank)
            with torch.no_grad():
                head.up_proj.weight.copy_(up)
                head.down_proj.weight.copy_(down)
        model.candidate_head = head
    model.candidate_head.validate_weights()
    model.requires_grad_(False)
    model.candidate_head.requires_grad_(True)
    model.candidate_head_only = True
    model.svd_candidate_head = None
    model.set_candidate_head(True, rerank=True)
    model.train(model.training)


class CandidateOnlyObjective:
    """Distill W h into A B h; no target logits or backbone loss are computed."""

    def __init__(
        self,
        model,
        docs,
        mask,
        decay,
        collect,
        *,
        stride=1,
        grad_addmm=False,
        backward_recompute=False,
        need_predictions=False,
    ):
        self.model, self.docs, self.decay = model, docs, decay
        self.valid = docs >= 0
        if mask is not None:
            self.valid = self.valid & mask.bool()
        self.stride, self.collect = stride, collect
        self.need_predictions = need_predictions
        self.accumulator = model.candidate_head.loss_accumulator(
            grad_addmm=grad_addmm,
            valid_only=True,
            backward_recompute=backward_recompute,
        )
        self.counts = {
            name: model.lm_head.weight.new_zeros((), dtype=torch.float32)
            for name in ("kl", "recall", "total")
        }

    def step(self, _logits, hidden, step):
        width = hidden.shape[1] - step
        if width > 0:
            valid = (
                self.valid[:, :width]
                & self.valid[:, step:]
                & (self.docs[:, :width] == self.docs[:, step:])
            )
            valid = select_candidate_positions(valid, self.stride, step)
            # Compact for both streamed and replay losses. Retain a masked row on
            # empty ranks so autograd/DDP still deliver zero parameter gradients.
            rows = hidden[:, :width].detach().reshape(-1, hidden.shape[-1])
            valid = valid.reshape(-1)
            indices = valid.nonzero(as_tuple=True)[0]
            selected = indices if indices.numel() else slice(0, 1)
            values = self.accumulator.add(
                rows[selected],
                self.model.lm_head.weight,
                valid[selected],
                top_k=self.model.config.candidate_head_top_k,
                chunk_size=self.model.config.candidate_head_chunk_size,
                normalize=True,
                loss_scale=self.decay**step,
                collect_recall=self.collect,
            )
            if self.collect:
                for name, value in zip(self.counts, values, strict=True):
                    self.counts[name] += value.detach()
        if self.need_predictions:
            with torch.no_grad():
                return predict_chunks(
                    hidden,
                    self.model.lm_head.weight,
                    self.model.config.candidate_head_chunk_size,
                )
        return None

    def finish(self, predictions):
        loss = self.accumulator.loss() * self.model.config.candidate_head_loss_weight
        metrics = {
            "loss_sum": loss.detach().clone(),
            "loss_total": loss.new_ones(()),
            "supervision_total": self.valid.sum(dtype=torch.float32),
        }
        if self.collect:
            metrics.update(
                candidate_kl_sum=self.counts["kl"],
                candidate_kl_total=self.counts["total"],
                candidate_recall_sum=self.counts["recall"],
                candidate_recall_total=self.counts["total"],
            )
        return predictions, loss, metrics
