"""Unit tests for the DSpark loss and metrics."""

import torch

from speculators.models.dspark.metrics import compute_metrics
from speculators.models.metrics import resolve_loss_config

_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeMetrics:
    def test_perfect_draft_low_loss_high_accept(self):
        # block_size=2; position 0 is the anchor (masked), position 1 supervised.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        assert torch.isfinite(loss)
        # Matching distributions -> CE/TV ~ 0 and acceptance ~ 1.
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # One draft slot per block accepted w.p. ~1, plus the anchor token -> ~2.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 2.0) < 1e-2

    def test_confidence_target_is_overlap(self):
        # When draft == target, accept rate == 1, so a confidence logit that is
        # very positive (sigmoid -> 1) yields ~zero abs error.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        abs_err = (
            metrics["confidence_abs_error_sum"] / metrics["confidence_abs_error_total"]
        )
        assert float(abs_err) < 1e-2
        assert "confidence_loss_sum" in metrics

    def test_confidence_term_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        # A badly-calibrated confidence head (predicts accept~1 when accept~0)
        # must add positive BCE on top of the base loss.
        confidence_logits = torch.full((1, 4), 20.0)
        loss_conf, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            confidence_head_alpha=1.0,
        )
        assert float(loss_conf) > float(loss_no_conf)

    def test_confidence_cumprod_bias_sign(self):
        # Draft != target so accept rate is ~0; an over-confident head (predicts
        # accept ~1) must show a positive cumulative-product calibration bias.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

    def test_confidence_loss_ignores_position_decay_and_cat(self):
        torch.manual_seed(5)
        logits = torch.randn(1, 4, 8)
        targets = torch.randn(1, 4, 8)
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        confidence_logits = torch.tensor([[0.0, -0.5, 0.5, 1.0]])

        _, plain_metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            gamma=1.0,
            cat_mode="none",
        )
        _, weighted_metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            gamma=100.0,
            cat_mode="draft",
        )
        assert torch.allclose(
            plain_metrics["confidence_loss_sum"],
            weighted_metrics["confidence_loss_sum"],
        )

    def test_confidence_draft_weighting_uses_one_active_weight(self):
        torch.manual_seed(6)
        logits = torch.randn(1, 4, 8)
        targets = torch.randn(1, 4, 8)
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        confidence_logits = torch.tensor([[0.0, -0.5, 0.5, 1.0]])

        _, uniform_metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            confidence_loss_weighting="uniform",
        )
        _, draft_metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            confidence_loss_weighting="draft",
        )
        assert not torch.allclose(
            uniform_metrics["confidence_loss_sum"],
            draft_metrics["confidence_loss_sum"],
        )

        _, cat_fast_decay = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            gamma=1.0,
            cat_mode="draft",
            confidence_loss_weighting="draft",
        )
        _, cat_slow_decay = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            gamma=100.0,
            cat_mode="draft",
            confidence_loss_weighting="draft",
        )
        assert torch.allclose(
            cat_fast_decay["confidence_loss_sum"],
            cat_slow_decay["confidence_loss_sum"],
        )

    def test_cat_replaces_fixed_decay_for_final_draft_loss(self):
        torch.manual_seed(7)
        logits = torch.randn(1, 4, 8)
        targets = torch.randn(1, 4, 8)
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])

        loss_fast_decay, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            gamma=1.0,
            cat_mode="draft",
        )
        loss_slow_decay, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            gamma=100.0,
            cat_mode="draft",
        )
        assert torch.allclose(loss_fast_decay, loss_slow_decay)

    def test_first_error_focal_targets_chain_breaker(self):
        logits = _ids_to_logits(torch.tensor([[0, 4, 5, 3]]), 8)
        targets = _ids_to_logits(torch.tensor([[0, 1, 2, 3]]), 8)
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])

        base_loss, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            first_error_focal_alpha=0.0,
        )
        focal_loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            first_error_focal_alpha=0.3,
        )
        focal_term = metrics["first_error_focal_loss_sum"]
        assert float(focal_term) > 0
        assert torch.allclose(focal_loss - base_loss, 0.3 * focal_term)
        mean_breaker = (
            metrics["first_error_position_sum"]
            / metrics["first_error_position_total"]
        )
        assert torch.isclose(mean_breaker, torch.tensor(1.0))

    def test_first_error_focal_is_zero_for_correct_block(self):
        logits = _ids_to_logits(torch.tensor([[0, 1, 2, 3]]), 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        _, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            first_error_focal_alpha=0.3,
        )
        assert torch.isclose(
            metrics["first_error_focal_loss_sum"], torch.tensor(0.0)
        )

    def test_curriculum_scales_first_error_focal_with_final_branch(self):
        final_logits = _ids_to_logits(torch.tensor([[0, 4, 2, 3]]), 8)
        targets = _ids_to_logits(torch.tensor([[0, 1, 2, 3]]), 8)
        base_logits = targets.clone()
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])

        without_focal, _ = compute_metrics(
            final_logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            base_logits=base_logits,
            curriculum_base_weight=1.0,
            first_error_focal_alpha=0.0,
        )
        with_focal, _ = compute_metrics(
            final_logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            base_logits=base_logits,
            curriculum_base_weight=1.0,
            first_error_focal_alpha=0.3,
        )
        assert torch.allclose(without_focal, with_focal)

    def test_confidence_length_loss_changes_total_loss(self):
        ids = torch.tensor([[0, 1, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0.0, 1.0, 1.0]])
        confidence_logits = torch.zeros(1, 3)

        loss_without, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=3,
            loss_config=_DEFAULT_LOSS,
            confidence_length_alpha=0.0,
        )
        loss_with, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=3,
            loss_config=_DEFAULT_LOSS,
            confidence_length_alpha=1.0,
        )
        length_loss = metrics["confidence_length_loss_sum"]
        assert float(length_loss) > 0
        assert torch.allclose(loss_with - loss_without, length_loss)
        assert "confidence_accept_len_pred_sum" in metrics

    def test_alpha_weighting(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_small, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 0.1}'),
        )
        loss_large, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 1.0}'),
        )
        assert float(loss_large) > float(loss_small)

    def test_metric_keys_present(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            torch.zeros(1, 4),
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        for key in (
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "tv_loss_sum",
            "full_acc_sum",
            "full_acc_total",
            "position_1_acc_sum",
            "accept_len_sum",
            "accept_len_total",
            "confidence_cumprod_bias_sum",
            "confidence_length_loss_sum",
            "confidence_accept_len_pred_sum",
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())

    def test_target_cat_changes_loss_and_logs_weight(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_none, metrics_none = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="none",
        )
        loss_cat, metrics_cat = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="target",
        )
        assert "cat_weight_mean_sum" not in metrics_none
        assert "cat_weight_mean_sum" in metrics_cat
        # With mismatched draft/target, CAT still produces a finite loss.
        assert torch.isfinite(loss_cat)
        assert torch.isfinite(loss_none)

    def test_draft_cat_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_none, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="none",
            sample_from_anchor=False,
        )
        loss_draft, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="draft",
            sample_from_anchor=False,
        )
        assert torch.isfinite(loss_draft)
        # Mismatched distributions -> low accept_rate -> later CAT weights < 1,
        # so draft-CAT loss should be <= unweighted loss for the same terms.
        assert float(loss_draft) <= float(loss_none) + 1e-5
        assert "cat_weight_mean_sum" in metrics

    def test_base_to_final_curriculum_endpoints(self):
        torch.manual_seed(4)
        targets = torch.randn(1, 4, 8)
        base_logits = targets.clone()
        final_logits = -targets
        loss_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
        config = resolve_loss_config("tv")

        base_loss, base_metrics = compute_metrics(
            final_logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=config,
            base_logits=base_logits,
            curriculum_base_weight=1.0,
        )
        final_loss, final_metrics = compute_metrics(
            final_logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=config,
            base_logits=base_logits,
            curriculum_base_weight=0.0,
        )
        assert float(base_loss) < float(final_loss)
        assert torch.isclose(
            base_metrics["curriculum_base_weight_sum"], torch.tensor(1.0)
        )
        assert torch.isclose(
            final_metrics["curriculum_base_weight_sum"], torch.tensor(0.0)
        )
        assert "base_loss_sum" in final_metrics
        assert "final_loss_sum" in final_metrics
