"""Unit tests for DSpark Markov and confidence heads."""

import pytest
import torch

from speculators.models.dspark.model_definitions import (
    CausalCorrectionHead,
    ConfidenceHead,
    MarkovHead,
)


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


class TestCausalCorrectionHead:
    def _head(self):
        torch.manual_seed(0)
        return CausalCorrectionHead(
            input_hidden_size=16,
            token_embedding_size=16,
            correction_hidden_size=16,
            correction_rank=8,
            draft_vocab_size=20,
            num_layers=2,
            num_heads=4,
        )

    def test_shape_and_zero_initialized_correction(self):
        head = self._head()
        token_emb = torch.randn(3, 5, 16)
        dflash_hidden = torch.randn(3, 5, 16)
        correction, states, cache = head(token_emb, dflash_hidden)
        assert correction.shape == (3, 5, 20)
        assert states.shape == (3, 5, 16)
        assert cache is None
        assert torch.count_nonzero(correction) == 0

    def test_causal_prefix_is_independent_of_future_inputs(self):
        head = self._head()
        token_emb = torch.randn(2, 5, 16)
        dflash_hidden = torch.randn(2, 5, 16)
        _, states_a, _ = head(token_emb, dflash_hidden)

        token_emb[:, 3:] = torch.randn_like(token_emb[:, 3:]) * 100
        dflash_hidden[:, 3:] = torch.randn_like(dflash_hidden[:, 3:]) * 100
        _, states_b, _ = head(token_emb, dflash_hidden)
        assert torch.allclose(states_a[:, :3], states_b[:, :3], atol=1e-5)

    def test_cached_steps_match_full_causal_forward(self):
        head = self._head().eval()
        token_emb = torch.randn(2, 5, 16)
        dflash_hidden = torch.randn(2, 5, 16)
        _, full_states, _ = head(token_emb, dflash_hidden)

        cache = None
        step_states = []
        for position in range(5):
            _, state, cache = head(
                token_emb[:, position : position + 1],
                dflash_hidden[:, position : position + 1],
                cache=cache,
                use_cache=True,
            )
            step_states.append(state)
        cached_states = torch.cat(step_states, dim=1)
        assert torch.allclose(full_states, cached_states, atol=1e-5)
        assert cache is not None
        assert all(key.shape[-2] == 5 for key, _ in cache)

    def test_misaligned_inputs_raise(self):
        head = self._head()
        with pytest.raises(ValueError, match="must align"):
            head(torch.randn(2, 4, 16), torch.randn(2, 5, 16))
