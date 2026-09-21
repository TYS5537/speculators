# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.
"""Standalone Markov decoding must preserve the training-time block recurrence."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    root = Path(__file__).parents[3]
    evaluator = _load_file(
        "dspark_markov_eval_test", root / "scripts/evaluate/dspark_offline_eval.py"
    )
    evaluator.torch = torch
    # Load the real torch-only heads file without importing the model package.
    heads = _load_file(
        "dspark_markov_heads_test",
        root / "src/speculators/models/dspark/model_definitions.py",
    )
    return evaluator, heads


def _runner(evaluator, head, sample_from_anchor, temperature):
    draft = SimpleNamespace(
        block_size=4,
        config=SimpleNamespace(sample_from_anchor=sample_from_anchor),
        correction_head=None,
        markov_head=head,
        candidate_selector=None,
        use_draft_vocab=False,
        d2t=None,
    )
    runner = evaluator.DSparkOfflineRunner.__new__(evaluator.DSparkOfflineRunner)
    runner.draft_model = draft
    runner.args = SimpleNamespace(temperature=temperature)
    runner.device = torch.device("cpu")
    runner.first_draft_slot = evaluator.first_draft_slot_for_draft(draft)
    runner.max_proposal_tokens = evaluator.speculative_slots_for_draft(draft)
    return runner


def _recurrent_head(heads):
    head = heads.MarkovHead(
        verifier_vocab_size=3,
        draft_vocab_size=3,
        markov_rank=1,
        hidden_size=1,
        head_type="rnn",
    )
    with torch.no_grad():
        head.markov_w1.weight.copy_(torch.tensor([[1.0], [0.25], [-1.0]]))
        head.markov_w2.weight.copy_(torch.tensor([[0.0], [1.0], [2.0]]))
        head.joint_proj.weight.zero_()
        head.joint_proj.bias.zero_()
        head.joint_proj.weight[1, 1] = 1.0  # Update state from the previous token.
        head.joint_proj.weight[2, 0] = 1.0  # Read that state on the next position.
    return head


@pytest.mark.parametrize("sample_from_anchor", [True, False])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_rnn_rollout_matches_full_block_with_real_generated_predecessors(
    modules, monkeypatch, sample_from_anchor, temperature
):
    evaluator, heads = modules
    head = _recurrent_head(heads)
    runner = _runner(evaluator, head, sample_from_anchor, temperature)
    monkeypatch.setattr(evaluator, "sample_from_probs", lambda probs: probs.argmax(-1))
    block_bias = head.block_bias
    histories = []

    def record_bias(*, prev_token_ids, hidden_states):
        histories.append(prev_token_ids.clone())
        assert hidden_states.shape[1] == prev_token_ids.shape[1]
        return block_bias(prev_token_ids=prev_token_ids, hidden_states=hidden_states)

    monkeypatch.setattr(head, "block_bias", record_bias)
    hidden = torch.zeros(1, 4, 1)
    with torch.no_grad():
        tokens, probabilities = runner._sample_dspark_tokens(
            torch.zeros(1, 4, 3), hidden, torch.tensor([1])
        )
        expected_tokens = [0, 2, 2, 0] if sample_from_anchor else [2, 2, 0]
        assert tokens == expected_tokens
        previous_ids = [1, *tokens[:-1]]
        if not sample_from_anchor:
            previous_ids.insert(0, 1)
        expected_logits = block_bias(
            prev_token_ids=torch.tensor([previous_ids]), hidden_states=hidden
        )[:, runner.first_draft_slot :]
        expected = evaluator.logits_to_probs(expected_logits, temperature)

    torch.testing.assert_close(probabilities, expected)
    assert [history.tolist()[0] for history in histories] == [
        previous_ids[:end] for end in range(runner.first_draft_slot + 1, 5)
    ]


@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_rnn_state_restarts_for_each_new_draft_block(modules, sample_from_anchor):
    evaluator, heads = modules
    runner = _runner(evaluator, _recurrent_head(heads), sample_from_anchor, 0.0)
    with torch.no_grad():
        first_tokens, first_probs = runner._sample_dspark_tokens(
            torch.zeros(1, 4, 3), torch.zeros(1, 4, 1), torch.tensor([1])
        )
        next_tokens, next_probs = runner._sample_dspark_tokens(
            torch.zeros(1, 4, 3), torch.zeros(1, 4, 1), torch.tensor([1])
        )

    assert next_tokens == first_tokens
    torch.testing.assert_close(next_probs, first_probs)


@pytest.mark.parametrize("head_type", ["vanilla", "gated"])
@pytest.mark.parametrize("sample_from_anchor", [True, False])
def test_nonrecurrent_markov_keeps_single_position_inference(
    modules, monkeypatch, head_type, sample_from_anchor
):
    evaluator, heads = modules
    torch.manual_seed(21)
    head = heads.MarkovHead(
        verifier_vocab_size=3,
        draft_vocab_size=3,
        markov_rank=2,
        hidden_size=2,
        head_type=head_type,
    )
    runner = _runner(evaluator, head, sample_from_anchor, 1.0)
    monkeypatch.setattr(evaluator, "sample_from_probs", lambda probs: probs.argmax(-1))
    block_bias = head.block_bias
    seen_lengths = []

    def record_bias(*, prev_token_ids, hidden_states):
        seen_lengths.append(prev_token_ids.shape[1])
        return block_bias(prev_token_ids=prev_token_ids, hidden_states=hidden_states)

    monkeypatch.setattr(head, "block_bias", record_bias)
    base_logits = torch.randn(1, 4, 3)
    hidden = torch.randn(1, 4, 2)
    with torch.no_grad():
        tokens, probabilities = runner._sample_dspark_tokens(
            base_logits, hidden, torch.tensor([1])
        )
        previous_ids = [1, *tokens[:-1]]
        active_hidden = hidden[:, runner.first_draft_slot :]
        expected_logits = base_logits[:, runner.first_draft_slot :] + block_bias(
            prev_token_ids=torch.tensor([previous_ids]), hidden_states=active_hidden
        )

    assert seen_lengths == [1] * runner.max_proposal_tokens
    assert tokens == expected_logits.argmax(-1)[0].tolist()
    torch.testing.assert_close(probabilities, expected_logits.softmax(-1))
