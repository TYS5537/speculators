"""Supervision, candidate distillation and metrics for one LEAGLE forward."""

import torch

from speculators.losses import kl_div_loss
from speculators.models.eagle3.metrics import EagleTrainingMetrics, compute_metrics
from speculators.models.leagle.chunked_loss import (
    chunked_training_step,
    predict_chunks,
    validate_chunked_loss,
)
from speculators.models.leagle.head_gradients import supports_fp32_head
from speculators.models.leagle.masked_loss import masked_training_step
from speculators.models.leagle.shared_loss import CandidateTeacher
from speculators.models.leagle.teacher_cache import TeacherCache, validate_teacher_cache
from speculators.models.leagle.training import TVOverlapProbe


class TrainingObjective:
    def __init__(
        self,
        model,
        teacher_hidden,
        docs,
        mask,
        steps,
        decay,
        loss_config,
        collect,
        chunk_size,
        initial_loss,
        *,
        grad_addmm=False,
        candidate_backward_recompute=False,
        teacher_cache_mb=0,
        masked_head=False,
        need_predictions=True,
        cache_supervised=False,
        shared_normalization=False,
        stream_main_loss=False,
    ):
        self.model, self.docs, self.decay = model, docs, decay
        # Combining CE/TV adjoints before a low-precision cast changes gradients.
        # Keep native loss kernels in BF16/FP16, including FP32-input autocast.
        self.shared_normalization = shared_normalization and supports_fp32_head(
            teacher_hidden
        )
        self.stream_main_loss = stream_main_loss
        self.candidate_backward_recompute = candidate_backward_recompute
        self.masked_head, self.need_predictions = masked_head, need_predictions
        self.collect_metrics, self.chunk_size = collect, chunk_size
        self.loss_config = (
            {"kl_div": (kl_div_loss, 1.0)} if loss_config is None else loss_config
        )
        self.loss = initial_loss
        self.candidate_loss = (
            model.candidate_head.loss_accumulator(
                grad_addmm=grad_addmm,
                valid_only=masked_head,
                backward_recompute=candidate_backward_recompute,
            )
            if model.candidate_head is not None
            else None
        )
        self.candidate_counts = {
            name: initial_loss.float().clone() for name in ("kl", "recall", "total")
        }
        self.valid = docs >= 0
        if mask is not None:
            self.valid = self.valid & mask.bool()
        self.metrics = EagleTrainingMetrics(docs, mask, steps) if collect else None
        if chunk_size:
            validate_chunked_loss(self.loss_config)
        if shared_normalization or stream_main_loss:
            validate_teacher_cache(self.loss_config, teacher_hidden.device)
        if stream_main_loss and (
            model.lm_head.weight.requires_grad
            or model.verifier_lm_head.weight.requires_grad
        ):
            raise ValueError("Streaming main loss requires frozen output matrices")
        with torch.no_grad():
            self.targets = model.verifier_norm(teacher_hidden)
            if not chunk_size:
                self.targets = model.verifier_lm_head(self.targets)
        self.teacher_cache = None
        if teacher_cache_mb:
            validate_teacher_cache(self.loss_config, teacher_hidden.device)
            self.teacher_cache = TeacherCache(
                self.targets,
                model.verifier_lm_head.weight,
                budget_bytes=teacher_cache_mb * 1024 * 1024,
                chunk_size=chunk_size,
                priority_mask=self.valid if cache_supervised else None,
                shared_normalization=self.shared_normalization,
            )
        self.target_ids = (
            self.targets.argmax(-1) if collect and not chunk_size else None
        )
        self.probe = (
            TVOverlapProbe(self.loss_config) if collect and not chunk_size else None
        )
        if self.probe is not None:
            self.loss_config = self.probe.loss_config

    def step(self, logits, hidden, step):
        if step >= hidden.shape[1]:
            if not self.need_predictions:
                return None
            return (
                predict_chunks(self.model, hidden, self.chunk_size)
                if logits is None
                else logits.argmax(-1)
            )
        mask = self.valid.clone()
        if step:
            mask[:, step:] &= self.docs[:, step:] == self.docs[:, :-step]
        ids, loss = (
            masked_training_step(self, hidden, step, mask)
            if self.masked_head
            else chunked_training_step(self, hidden, step, mask)
            if self.chunk_size
            else self._dense_step(logits, hidden, step, mask)
        )
        self.loss = self.loss + loss
        if self.candidate_backward_recompute:
            # A complete TTT step owns one candidate backward, independently of
            # the main loss chunks/compaction. Do not retain their dense logits.
            width = hidden.shape[1] - step
            counts = self.candidate_loss.add(
                hidden[:, :width],
                self.model.lm_head.weight,
                self.valid[:, :width] & mask[:, step:],
                top_k=self.model.config.candidate_head_top_k,
                chunk_size=self.model.config.candidate_head_chunk_size,
                normalize=True,
                loss_scale=self.decay**step,
                collect_recall=self.collect_metrics,
            )
            self._record_candidate(counts)
        return ids

    def _dense_step(self, logits, hidden, step, mask):
        ids = (
            logits.argmax(-1) if self.need_predictions or self.collect_metrics else None
        )
        loss, _ = compute_metrics(
            logits,
            self.targets,
            mask,
            None,
            step,
            self.decay,
            loss_config=self.loss_config,
            collect_metrics=False,
        )
        if self.metrics is not None:
            self.metrics.update(
                logits,
                self.targets,
                step,
                draft_ids=ids,
                target_ids=self.target_ids,
                overlap=self.probe.overlap,
            )
        width = hidden.shape[1] - step
        self.add_candidate(
            hidden[:, :width],
            logits[:, :width],
            self.valid[:, :width] & mask[:, step:],
            normalize=True,
            loss_scale=self.decay**step,
        )
        return ids if self.need_predictions else None, loss

    def add_candidate(self, hidden, logits, mask, *, loss_scale, normalize=False):
        if self.candidate_loss is None or self.candidate_backward_recompute:
            return
        counts = self.candidate_loss.add(
            hidden,
            self.model.lm_head.weight,
            mask,
            top_k=self.model.config.candidate_head_top_k,
            chunk_size=self.model.config.candidate_head_chunk_size,
            teacher_logits=None if isinstance(logits, CandidateTeacher) else logits,
            teacher_distribution=logits
            if isinstance(logits, CandidateTeacher)
            else None,
            normalize=normalize,
            loss_scale=loss_scale,
            collect_recall=self.collect_metrics,
        )
        self._record_candidate(counts)

    def _record_candidate(self, counts):
        if self.collect_metrics:
            for name, value in zip(("kl", "recall", "total"), counts, strict=True):
                self.candidate_counts[name] += value.detach()

    def finish(self, predictions):
        total_loss = self.loss
        if self.candidate_loss is not None:
            total_loss = (
                total_loss
                + self.model.config.candidate_head_loss_weight
                * self.candidate_loss.loss()
            )
        counts = (
            self.metrics.finish(total_loss)
            if self.metrics is not None
            else {
                "loss_sum": total_loss.detach().clone(),
                "loss_total": total_loss.new_ones(()),
            }
        )
        if self.metrics is not None and self.candidate_loss is not None:
            counts.update(
                {
                    "draft_loss_sum": self.loss.detach(),
                    "draft_loss_total": self.loss.new_ones(()),
                    "candidate_kl_sum": self.candidate_counts["kl"],
                    "candidate_kl_total": self.candidate_counts["total"],
                    "candidate_recall_sum": self.candidate_counts["recall"],
                    "candidate_recall_total": self.candidate_counts["total"],
                }
            )
        return predictions, total_loss, counts
