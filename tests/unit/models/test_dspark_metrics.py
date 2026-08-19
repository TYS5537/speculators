"""Unit tests for the DSpark loss and metrics."""

import torch

from speculators.models.dspark.metrics import compute_metrics, select_logged_metrics
from speculators.models.metrics import resolve_loss_config

_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeMetrics:
    def test_logged_metrics_use_compact_shared_schema(self):
        metrics = {
            "loss_sum": torch.tensor(1.0),
            "loss_total": torch.tensor(1.0),
            "full_acc_sum": torch.tensor(1.0),
            "accept_len_sum": torch.tensor(1.0),
            "position_0_acc_sum": torch.tensor(1.0),
            "confidence_loss_sum": torch.tensor(1.0),
            "correction_hidden_aux_loss_sum": torch.tensor(1.0),
            "dflash2_selector_loss_sum": torch.tensor(1.0),
            "collaboration_accept_len_gain_sum": torch.tensor(1.0),
            "collaboration_markov_gate_mean_sum": torch.tensor(1.0),
            "collaboration_markov_change_accuracy_sum": torch.tensor(1.0),
            "collaboration_markov_harmed_count_sum": torch.tensor(1.0),
            "rollout_full_acc_sum": torch.tensor(1.0),
            "rollout_accept_len_sum": torch.tensor(1.0),
            "accept_rate_sum": torch.tensor(1.0),
            "ce_loss_sum": torch.tensor(1.0),
            "correction_logit_rms_sum": torch.tensor(1.0),
            "collaboration_markov_change_wrong_count_sum": torch.tensor(1.0),
        }

        selected = select_logged_metrics(metrics)

        assert set(selected) == {
            "loss_sum",
            "loss_total",
            "full_acc_sum",
            "accept_len_sum",
            "position_0_acc_sum",
            "confidence_loss_sum",
            "correction_hidden_aux_loss_sum",
            "dflash2_selector_loss_sum",
            "collaboration_accept_len_gain_sum",
            "collaboration_markov_gate_mean_sum",
            "collaboration_markov_change_accuracy_sum",
            "collaboration_markov_harmed_count_sum",
            "rollout_full_acc_sum",
            "rollout_accept_len_sum",
        }
        assert select_logged_metrics(metrics, include_diagnostics=True) is metrics

    def test_perfect_draft_low_loss_high_accept(self):
        # block_size=2; with sample_from_anchor=False, position 0 is the anchor
        # (masked) and position 1 supervised.
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
            sample_from_anchor=False,
        )
        assert torch.isfinite(loss)
        # Matching distributions -> CE/TV ~ 0 and acceptance ~ 1.
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # One draft slot per block accepted w.p. ~1, plus the anchor token -> ~2.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 2.0) < 1e-2

    def test_perfect_draft_anchor_sampled_includes_slot0(self):
        # sample_from_anchor=True (default): slot 0 is the first real prediction,
        # so every position is supervised and accept_len counts all draft slots.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.ones(1, 4, dtype=torch.float32)
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
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # Two draft slots per block accepted w.p. ~1, plus the anchor token -> ~3.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 3.0) < 1e-2

    def test_selector_distribution_drives_acceptance_and_accuracy_metrics(self):
        target_ids = torch.tensor([[1, 1]])
        logits = _ids_to_logits(target_ids, 4)
        targets = logits.clone()
        loss_mask = torch.ones(1, 2)
        candidate_ids = torch.tensor([[[0, 1], [0, 1]]])
        candidate_logits = torch.tensor([[[100.0, 0.0], [100.0, 0.0]]])

        _, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            proposal_candidate_ids=candidate_ids,
            proposal_candidate_logits=candidate_logits,
        )

        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) < 1e-4
        assert metrics["full_acc_sum"] == 0

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
        # sample_from_anchor=False: position 0 is the anchor (masked).
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
            sample_from_anchor=False,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

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
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())

    def test_ssal_ignores_gamma(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 8)
        targets = torch.randn(1, 4, 8)
        loss_mask = torch.ones(1, 4)
        loss_a, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            adaptive_loss="ssal",
            gamma=1.0,
        )
        loss_b, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            adaptive_loss="ssal",
            gamma=100.0,
        )
        assert torch.allclose(loss_a, loss_b)

    def test_ssal_curriculum_mix_between_decay_and_ssal(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 4, 8)
        targets = torch.randn(1, 4, 8)
        loss_mask = torch.ones(1, 4)
        pure_decay, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            adaptive_loss="none",
            gamma=4.0,
        )
        pure_ssal, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            adaptive_loss="ssal",
            ssal_decay_weight=0.0,
        )
        mixed, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            adaptive_loss="ssal",
            ssal_decay_weight=1.0,
            gamma=4.0,
        )
        assert torch.allclose(mixed, pure_decay)
        assert not torch.allclose(pure_ssal, pure_decay)

    def test_correction_gain_metrics(self):
        target_ids = torch.tensor([[1, 2, 3, 4]])
        targets = _ids_to_logits(target_ids, 8)
        corrected_logits = targets.clone()
        base_logits = _ids_to_logits(torch.tensor([[5, 6, 7, 0]]), 8)
        loss_mask = torch.ones(1, 4)

        corrected_loss, corrected_metrics = compute_metrics(
            corrected_logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            base_logits=base_logits,
        )

        assert float(corrected_loss) < float(corrected_metrics["base_loss_sum"])
        gain = (
            corrected_metrics["correction_accept_len_gain_sum"]
            / corrected_metrics["correction_accept_len_gain_total"]
        )
        assert float(gain) > 1.5
        correction_rms = (
            corrected_metrics["correction_logit_rms_sum"]
            / corrected_metrics["correction_logit_rms_total"]
        )
        assert float(correction_rms) > 0.0
        argmax_change = (
            corrected_metrics["correction_argmax_change_rate_sum"]
            / corrected_metrics["correction_argmax_change_rate_total"]
        )
        assert float(argmax_change) == 1.0
        assert float(corrected_metrics["dspark_head_change_correct_count_sum"]) == 4.0
        assert float(corrected_metrics["dspark_head_change_wrong_count_sum"]) == 0.0
        assert float(corrected_metrics["dspark_head_harmed_count_sum"]) == 0.0

    def test_head_and_rollout_change_outcome_counts(self):
        targets = _ids_to_logits(torch.tensor([[1, 2, 3, 4]]), 8)
        base_logits = _ids_to_logits(torch.tensor([[1, 6, 7, 4]]), 8)
        head_logits = _ids_to_logits(torch.tensor([[5, 2, 0, 4]]), 8)
        rollout_logits = _ids_to_logits(torch.tensor([[1, 2, 0, 5]]), 8)

        _, metrics = compute_metrics(
            head_logits,
            targets,
            None,
            torch.ones(1, 4),
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            base_logits=base_logits,
            rollout_logits=rollout_logits,
        )

        assert float(metrics["dspark_head_change_correct_count_sum"]) == 1.0
        assert float(metrics["dspark_head_change_wrong_count_sum"]) == 2.0
        assert float(metrics["dspark_head_harmed_count_sum"]) == 1.0
        assert float(metrics["dspark_head_change_accuracy_sum"]) == 1.0
        assert float(metrics["dspark_head_change_accuracy_total"]) == 3.0
        assert float(metrics["causal_rollout_change_correct_count_sum"]) == 1.0
        assert float(metrics["causal_rollout_change_wrong_count_sum"]) == 2.0
        assert float(metrics["causal_rollout_harmed_count_sum"]) == 1.0

    def test_rollout_metrics_are_separate_from_teacher_forcing(self):
        target_ids = torch.tensor([[1, 2, 3, 4]])
        targets = _ids_to_logits(target_ids, 8)
        teacher_forced_logits = targets.clone()
        rollout_logits = _ids_to_logits(torch.tensor([[1, 6, 7, 0]]), 8)
        loss_mask = torch.ones(1, 4)

        _, metrics = compute_metrics(
            teacher_forced_logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            rollout_logits=rollout_logits,
        )

        teacher_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        rollout_len = (
            metrics["rollout_accept_len_sum"] / metrics["rollout_accept_len_total"]
        )
        assert float(teacher_len) > float(rollout_len)
        assert float(teacher_len) > 2.9
        assert float(rollout_len) < 2.1

    def test_first_error_focal_increases_loss(self):
        logits = _ids_to_logits(torch.tensor([[0, 4, 5, 3]]), 8)
        targets = _ids_to_logits(torch.tensor([[0, 1, 2, 3]]), 8)
        loss_mask = torch.ones(1, 4)
        base, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            first_error_focal_alpha=0.0,
        )
        focal, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            first_error_focal_alpha=1.0,
        )
        assert float(focal) > float(base)

    def test_collaboration_metrics_compare_against_correction_only(self):
        target_ids = torch.tensor([[1, 2, 3, 4]])
        targets = _ids_to_logits(target_ids, 8)
        joint_logits = targets.clone()
        correction_only_logits = _ids_to_logits(torch.tensor([[0, 2, 0, 4]]), 8)
        loss_mask = torch.ones(1, 4)
        collaboration_gate = torch.full((2, 2, 1), 0.25)

        _, metrics = compute_metrics(
            joint_logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            collaboration_base_logits=correction_only_logits,
            collaboration_gate=collaboration_gate,
        )

        accept_gain = (
            metrics["collaboration_accept_len_gain_sum"]
            / metrics["collaboration_accept_len_gain_total"]
        )
        gate_mean = (
            metrics["collaboration_markov_gate_mean_sum"]
            / metrics["collaboration_markov_gate_mean_total"]
        )
        assert float(accept_gain) > 1.0
        assert abs(float(gate_mean) - 0.25) < 1e-6
        assert float(metrics["collaboration_markov_change_correct_count_sum"]) == 2.0
        assert float(metrics["collaboration_markov_change_wrong_count_sum"]) == 0.0

    def test_confidence_length_and_uniform_vs_match_draft(self):
        torch.manual_seed(2)
        logits = torch.randn(1, 4, 8)
        targets = torch.randn(1, 4, 8)
        loss_mask = torch.ones(1, 4)
        conf = torch.randn(1, 4)
        _, m_len = compute_metrics(
            logits,
            targets,
            conf,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            confidence_length_alpha=0.1,
        )
        assert "confidence_length_loss_sum" in m_len

        loss_u, _ = compute_metrics(
            logits,
            targets,
            conf,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            confidence_loss_weighting="uniform",
            gamma=1.0,
        )
        loss_m, _ = compute_metrics(
            logits,
            targets,
            conf,
            loss_mask,
            block_size=4,
            loss_config=_DEFAULT_LOSS,
            confidence_loss_weighting="match-draft",
            gamma=1.0,
        )
        assert not torch.allclose(loss_u, loss_m)
