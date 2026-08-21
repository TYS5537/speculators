"""Unit tests for DSpark sequential and confidence heads."""

import pytest
import torch

from speculators.models.dspark.model_definitions import (
    CausalCorrectionHead,
    ConfidenceHead,
    MarkovHead,
)


class TestCausalCorrectionHead:
    def _head(self):
        torch.manual_seed(0)
        return CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            block_size=4,
            correction_hidden_size=12,
            correction_rank=8,
            num_layers=2,
            num_heads=3,
        ).eval()

    def _inputs(self, seq_len=4):
        torch.manual_seed(1)
        return (
            torch.randn(2, seq_len, 16),
            torch.randn(2, seq_len, 16),
            torch.arange(seq_len).expand(2, -1),
        )

    def test_shape_and_zero_initialized_residual(self):
        head = self._head()
        delta, states, cache = head(*self._inputs())
        assert delta.shape == (2, 4, 16)
        assert states.shape == (2, 4, 12)
        assert cache is None
        assert torch.count_nonzero(delta) == 0
        assert torch.isfinite(states).all()

    def test_current_selector_token_embedding_changes_correction_state(self):
        head = self._head()
        previous, hidden, positions = self._inputs()
        _, states_without_selector, _ = head(previous, hidden, positions)
        _, states_with_selector, _ = head(
            previous,
            hidden,
            positions,
            current_token_embeddings=torch.randn_like(hidden),
        )

        assert not torch.allclose(states_without_selector, states_with_selector)

    def test_hidden_residual_lm_head_fusion_matches_unfused_projection(self):
        torch.manual_seed(10)
        head = CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            block_size=4,
            correction_hidden_size=12,
            correction_rank=8,
            num_layers=1,
            num_heads=3,
        ).eval()
        with torch.no_grad():
            head.correction_up.weight.normal_()

            states = torch.randn(2, 3, 12)
            lm_head_weight = torch.randn(20, 16)
            expected = torch.nn.functional.linear(
                head.auxiliary_hidden_residual(states),
                lm_head_weight,
            )
            actual = head.fused_lm_head_residual(states, lm_head_weight)

            assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)

            # In-place parameter updates invalidate the lazy inference cache.
            head.correction_up.weight.add_(0.1)
            refreshed_expected = torch.nn.functional.linear(
                head.auxiliary_hidden_residual(states),
                lm_head_weight,
            )
            refreshed_actual = head.fused_lm_head_residual(states, lm_head_weight)
            assert torch.allclose(
                refreshed_actual,
                refreshed_expected,
                atol=1e-4,
                rtol=1e-4,
            )

    def test_logit_dual_hidden_residual_lm_head_fusion_matches_projection(self):
        torch.manual_seed(11)
        head = CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            block_size=4,
            correction_hidden_size=12,
            correction_rank=8,
            num_layers=1,
            num_heads=3,
            output_mode="logits",
            draft_vocab_size=20,
            enable_hidden_auxiliary=True,
        ).eval()
        assert head.auxiliary_hidden_up is not None
        with torch.no_grad():
            head.auxiliary_hidden_up.weight.normal_()

            states = torch.randn(2, 3, 12)
            lm_head_weight = torch.randn(20, 16)
            expected = torch.nn.functional.linear(
                head.auxiliary_hidden_residual(states),
                lm_head_weight,
            )
            actual = head.fused_lm_head_residual(states, lm_head_weight)

        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)

    def test_future_inputs_do_not_change_prefix(self):
        head = self._head()
        inputs = list(self._inputs())
        _, states_a, _ = head(*inputs)
        changed = [value.clone() for value in inputs]
        for value in changed[:2]:
            value[:, 2:] = torch.randn_like(value[:, 2:]) * 100.0
        changed[2][:, 2:] = torch.flip(changed[2][:, 2:], dims=(1,))
        _, states_b, _ = head(*changed)
        assert torch.allclose(states_a[:, :2], states_b[:, :2], atol=1e-5)

    def test_block_position_embedding_changes_states(self):
        head = self._head()
        previous, hidden, positions = self._inputs()
        _, states_a, _ = head(previous, hidden, positions)
        _, states_b, _ = head(previous, hidden, torch.zeros_like(positions))
        assert not torch.allclose(states_a[:, 1:], states_b[:, 1:])

    def test_cached_rollout_matches_full_sequence(self):
        head = self._head()
        inputs = self._inputs()
        _, full_states, _ = head(*inputs)

        cache = None
        step_states = []
        for position in range(inputs[0].shape[1]):
            step_inputs = tuple(value[:, position : position + 1] for value in inputs)
            _, states, cache = head(
                *step_inputs,
                cache=cache,
                use_cache=True,
            )
            step_states.append(states)

        assert cache is not None
        assert len(cache) == 2
        assert torch.allclose(
            full_states,
            torch.cat(step_states, dim=1),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_logit_residual_shape_and_previous_logit_feature(self):
        torch.manual_seed(2)
        head = CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            block_size=4,
            correction_hidden_size=12,
            correction_rank=8,
            num_layers=2,
            num_heads=3,
            output_mode="logits",
            draft_vocab_size=20,
        ).eval()
        previous, hidden, positions = self._inputs()
        previous_logits = torch.randn(2, 4, 20)
        previous_logits_mask = positions > 0

        delta, states_a, _ = head(
            previous,
            hidden,
            positions,
            previous_logits=previous_logits,
            previous_logits_mask=previous_logits_mask,
        )
        assert delta.shape == (2, 4, 20)
        assert states_a.shape == (2, 4, 12)
        assert torch.count_nonzero(delta) == 0

        # Position zero explicitly masks the feature, while later positions
        # consume the previous target distribution from the first step onward.
        changed_logits = previous_logits.clone()
        changed_logits[:, 0] = torch.randn_like(changed_logits[:, 0]) * 100.0
        _, states_b, _ = head(
            previous,
            hidden,
            positions,
            previous_logits=changed_logits,
            previous_logits_mask=previous_logits_mask,
        )
        assert torch.allclose(states_a[:, 0], states_b[:, 0], atol=1e-5)

        changed_logits[:, 1:] = torch.randn_like(changed_logits[:, 1:]) * 100.0
        _, states_c, _ = head(
            previous,
            hidden,
            positions,
            previous_logits=changed_logits,
            previous_logits_mask=previous_logits_mask,
        )
        assert not torch.allclose(states_b[:, 1:], states_c[:, 1:])

    def test_sparse_previous_distribution_matches_dense_topk_features(self):
        torch.manual_seed(13)
        head = CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            block_size=4,
            correction_hidden_size=12,
            correction_rank=8,
            num_layers=1,
            num_heads=3,
            draft_vocab_size=20,
            output_mode="logits",
        ).eval()
        candidate_ids = torch.tensor([[[1, 4, 7], [2, 5, 8]]])
        candidate_logits = torch.randn(1, 2, 3)
        mask = torch.tensor([[False, True]])
        dense_logits = torch.full((1, 2, 20), -torch.inf)
        dense_logits.scatter_(-1, candidate_ids, candidate_logits)

        dense_rank = head.encode_previous_distribution(
            mask,
            previous_logits=dense_logits,
        )
        sparse_rank = head.encode_previous_distribution(
            mask,
            candidate_ids=candidate_ids,
            candidate_logits=candidate_logits,
        )

        assert torch.allclose(dense_rank, sparse_rank, atol=1e-6)

    def test_logit_residual_cached_rollout_matches_full_sequence(self):
        torch.manual_seed(3)
        head = CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            block_size=4,
            correction_hidden_size=12,
            correction_rank=8,
            num_layers=2,
            num_heads=3,
            output_mode="logits",
            draft_vocab_size=20,
        ).eval()
        previous, hidden, positions = self._inputs()
        previous_logits = torch.randn(2, 4, 20)
        previous_logits_mask = positions > 0
        full_delta, full_states, _ = head(
            previous,
            hidden,
            positions,
            previous_logits=previous_logits,
            previous_logits_mask=previous_logits_mask,
        )

        cache = None
        step_delta = []
        step_states = []
        for position in range(previous.shape[1]):
            delta, states, cache = head(
                previous[:, position : position + 1],
                hidden[:, position : position + 1],
                positions[:, position : position + 1],
                previous_logits=previous_logits[:, position : position + 1],
                previous_logits_mask=previous_logits_mask[:, position : position + 1],
                cache=cache,
                use_cache=True,
            )
            step_delta.append(delta)
            step_states.append(states)

        assert torch.allclose(
            full_delta, torch.cat(step_delta, dim=1), atol=1e-5, rtol=1e-5
        )
        assert torch.allclose(
            full_states, torch.cat(step_states, dim=1), atol=1e-5, rtol=1e-5
        )

    def test_logit_mode_auxiliary_hidden_and_feedback_are_opt_in(self):
        torch.manual_seed(4)
        head = CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            block_size=4,
            correction_hidden_size=12,
            correction_rank=8,
            num_layers=1,
            num_heads=3,
            output_mode="logits",
            draft_vocab_size=20,
            enable_hidden_auxiliary=True,
            enable_hidden_feedback=True,
        ).eval()
        previous, hidden, positions = self._inputs()
        previous_logits = torch.randn(2, 4, 20)
        previous_logits_mask = positions > 0
        previous_corrected_hidden = torch.randn_like(hidden)
        previous_corrected_hidden_mask = positions > 0

        delta_logits, states_a, _ = head(
            previous,
            hidden,
            positions,
            previous_logits=previous_logits,
            previous_logits_mask=previous_logits_mask,
            previous_corrected_hidden=previous_corrected_hidden,
            previous_corrected_hidden_mask=previous_corrected_hidden_mask,
        )
        delta_hidden = head.auxiliary_hidden_residual(states_a)
        assert delta_logits.shape == (2, 4, 20)
        assert delta_hidden.shape == hidden.shape
        assert torch.count_nonzero(delta_hidden) == 0

        changed_hidden = previous_corrected_hidden.clone()
        changed_hidden[:, 0] = torch.randn_like(changed_hidden[:, 0]) * 100.0
        _, states_b, _ = head(
            previous,
            hidden,
            positions,
            previous_logits=previous_logits,
            previous_logits_mask=previous_logits_mask,
            previous_corrected_hidden=changed_hidden,
            previous_corrected_hidden_mask=previous_corrected_hidden_mask,
        )
        assert torch.allclose(states_a[:, 0], states_b[:, 0], atol=1e-5)

        changed_hidden[:, 1:] = torch.randn_like(changed_hidden[:, 1:]) * 100.0
        _, states_c, _ = head(
            previous,
            hidden,
            positions,
            previous_logits=previous_logits,
            previous_logits_mask=previous_logits_mask,
            previous_corrected_hidden=changed_hidden,
            previous_corrected_hidden_mask=previous_corrected_hidden_mask,
        )
        assert not torch.allclose(states_b[:, 1:], states_c[:, 1:])


class TestMarkovHead:
    def _head(self, head_type="vanilla", r=8, vv=50, dv=20, h=16):
        torch.manual_seed(0)
        return MarkovHead(
            verifier_vocab_size=vv,
            draft_vocab_size=dv,
            markov_rank=r,
            hidden_size=h,
            head_type=head_type,
        )

    @pytest.mark.parametrize("head_type", ["vanilla", "gated", "rnn"])
    def test_block_bias_shape(self, head_type):
        head = self._head(head_type)
        n, b, h = 3, 4, 16
        prev = torch.randint(0, 50, (n, b))
        hidden = torch.randn(n, b, h)
        bias = head.block_bias(prev_token_ids=prev, hidden_states=hidden)
        assert bias.shape == (n, b, 20)
        assert torch.isfinite(bias).all()

    def test_vanilla_is_low_rank_factorization(self):
        head = self._head("vanilla")
        prev = torch.randint(0, 50, (2, 4))
        hidden = torch.zeros(2, 4, 16)
        bias = head.block_bias(prev_token_ids=prev, hidden_states=hidden)
        expected = head.markov_w2(head.markov_w1(prev))
        assert torch.allclose(bias, expected, atol=1e-5)

    def test_bias_depends_on_prev_token(self):
        head = self._head("vanilla")
        hidden = torch.zeros(1, 1, 16)
        bias_a = head.block_bias(
            prev_token_ids=torch.tensor([[1]]), hidden_states=hidden
        )
        bias_b = head.block_bias(
            prev_token_ids=torch.tensor([[2]]), hidden_states=hidden
        )
        assert not torch.allclose(bias_a, bias_b)

    def test_invalid_rank_raises(self):
        with pytest.raises(ValueError):
            MarkovHead(
                verifier_vocab_size=50,
                draft_vocab_size=20,
                markov_rank=0,
                hidden_size=16,
            )

    def test_invalid_head_type_raises(self):
        with pytest.raises(ValueError):
            MarkovHead(
                verifier_vocab_size=50,
                draft_vocab_size=20,
                markov_rank=8,
                hidden_size=16,
                head_type="bogus",
            )


class TestConfidenceHead:
    def test_output_shape(self):
        head = ConfidenceHead(input_dim=24)
        features = torch.randn(3, 4, 24)
        out = head(features)
        assert out.shape == (3, 4)
        assert torch.isfinite(out).all()
