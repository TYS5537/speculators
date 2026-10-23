"""Acceptance correctness, target rollback and real LEAGLE checkpoint evaluation."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.leagle import LEagleDraftModel, LEagleSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators_eval import leagle, parallel
from speculators_eval.leagle_decoding import check_greedy_output, generate_greedy
from speculators_eval.reporting import EvalStats, aggregate_rows, summary_row


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


class CountingTarget(torch.nn.Module):
    """Predict n+1, recording the full cache so rollback errors cannot hide."""

    def __init__(self):
        super().__init__()
        self.config = Qwen3Config(num_hidden_layers=1, layer_types=["full_attention"])
        self.calls = []

    def forward(self, input_ids, position_ids, past_key_values, **kwargs):
        start = past_key_values.get_seq_length()
        assert position_ids[0, 0].item() == start
        if start:
            # Every processed token before the known anchor must be correct.
            cached = past_key_values.layers[0].keys.flatten().long()
            torch.testing.assert_close(cached, torch.arange(start) % 64)
        keys = input_ids.float().reshape(1, 1, -1, 1)
        past_key_values.update(keys, keys, 0)
        self.calls.append((start, input_ids.clone()))
        logits = torch.full((1, input_ids.shape[1], 64), -20.0)
        logits.scatter_(-1, ((input_ids + 1) % 64).unsqueeze(-1), 20.0)
        hidden = input_ids.float().unsqueeze(-1)
        return SimpleNamespace(logits=logits, hidden_states=(hidden, hidden + 100))


class ControlledDraft(torch.nn.Module):
    target_layer_ids = [0, 1]

    def __init__(self, reject_at=None, rejected_id=60):
        super().__init__()
        self.reject_at = reject_at
        self.rejected_id = rejected_id
        self.calls = []

    def propose(self, hidden_states, anchor_token, position_ids, num_tokens):
        length = hidden_states.shape[1]
        # Target features of rejected tokens must never enter the next round.
        torch.testing.assert_close(
            hidden_states[0, :, 0], torch.arange(length).float() % 64
        )
        torch.testing.assert_close(
            hidden_states[0, :, 1], (torch.arange(length).float() % 64) + 100
        )
        torch.testing.assert_close(position_ids, torch.arange(length).unsqueeze(0))
        self.calls.append((length, anchor_token.clone(), num_tokens))
        result = (anchor_token + torch.arange(1, num_tokens + 1)) % 64
        if self.reject_at is not None and self.reject_at < num_tokens:
            result[:, self.reject_at] = self.rejected_id
        return result, None


@pytest.mark.parametrize("reject_at", [None, 0, 2])
@pytest.mark.parametrize("budget", [0, 1, 2, 3, 9, 17])
def test_rounds_rejection_rollback_and_generation_budget(reject_at, budget):
    target, draft = CountingTarget().eval(), ControlledDraft(reject_at).eval()
    response = generate_greedy(
        target, draft, torch.tensor([[0, 1]]), max_new_tokens=budget
    )
    torch.testing.assert_close(
        response.output_ids, torch.arange(budget + 2).unsqueeze(0)
    )
    assert response.num_output_tokens == budget
    assert all(
        0 <= accepted <= proposed <= 7
        for accepted, proposed in zip(
            response.accepted_draft_lengths, response.proposal_lengths, strict=True
        )
    )
    stats = EvalStats(probability_diagnostics_available=False)
    stats.add_response(response)
    assert 1 <= stats.acceptance_length <= 8
    if budget == 17 and reject_at is None:
        assert response.proposal_lengths == response.accepted_draft_lengths == [7, 7]
        assert stats.acceptance_length == 8
    if budget == 2:
        assert not draft.calls  # Only one target token can fit after the anchor.
        assert response.proposal_lengths == [0]


@pytest.mark.parametrize("stop", [2, 3, 7, 10])
@pytest.mark.parametrize("reject_at", [None, 0, 2])
def test_eos_at_anchor_accepted_draft_and_bonus(stop, reject_at):
    response = generate_greedy(
        CountingTarget().eval(),
        ControlledDraft(reject_at).eval(),
        torch.tensor([[0, 1]]),
        max_new_tokens=24,
        stop_token_ids=[stop, 63],
    )
    torch.testing.assert_close(response.output_ids, torch.arange(stop + 1).unsqueeze(0))
    if stop == 2:
        assert response.proposal_lengths == []
    if stop == 7 and reject_at is None:
        assert response.proposal_lengths == response.accepted_draft_lengths == [5]


def test_rejected_eos_is_not_a_stop_and_first_mismatch_ends_acceptance():
    response = generate_greedy(
        CountingTarget().eval(),
        ControlledDraft(2, rejected_id=63).eval(),
        torch.tensor([[0, 1]]),
        max_new_tokens=17,
        stop_token_ids=[63],
    )
    torch.testing.assert_close(response.output_ids, torch.arange(19).unsqueeze(0))
    assert response.accepted_draft_lengths[0] == 2
    assert response.proposal_lengths[0] == 7


def make_real_models(layers=1, loops=1, reduced=True):
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=256,
        attention_dropout=0.0,
        layer_types=["full_attention"] * 3,
        eos_token_id=None,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(config).eval()
    decoder = copy.deepcopy(config)
    decoder.num_hidden_layers = layers
    decoder.layer_types = ["full_attention"] * layers
    draft = LEagleDraftModel(
        LEagleSpeculatorConfig(
            transformer_layer_config=decoder,
            draft_vocab_size=16 if reduced else 32,
            aux_hidden_state_layer_ids=[1, 2],
            num_loops=loops,
            speculators_config=SpeculatorsConfig(
                algorithm="leagle",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=7)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
    ).eval()
    selected = torch.arange(0, 32, 2) if reduced else torch.arange(32)
    if reduced:
        mask = torch.zeros(32, dtype=torch.bool)
        mask[selected] = True
        draft.load_vocab_mappings(mask, selected - torch.arange(16))
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(target.model.embed_tokens.weight)
        draft.lm_head.weight.copy_(target.lm_head.weight[selected])
    return target, draft


@pytest.mark.parametrize(("layers", "loops"), [(1, 1), (1, 3), (2, 2)])
@pytest.mark.parametrize("reduced", [False, True])
def test_real_qwen3_with_leagle_matches_independent_hf_generation(
    layers, loops, reduced
):
    target, draft = make_real_models(layers, loops, reduced)
    inputs = torch.tensor([[5, 9, 2]])
    response = generate_greedy(target, draft, inputs, max_new_tokens=23)
    check_greedy_output(target, inputs, response, 23, [])
    assert len(response.proposal_lengths) > 1
    # Evaluation uses the saved loop count and creates no new physical layers.
    assert draft.config.num_loops == loops
    assert len(draft.layers) == layers


def test_check_greedy_detects_incorrect_output():
    target, draft = make_real_models()
    inputs = torch.tensor([[5, 9, 2]])
    response = generate_greedy(target, draft, inputs, max_new_tokens=4)
    response.output_ids = response.output_ids.clone()
    response.output_ids[0, -1] = (response.output_ids[0, -1] + 1) % 32
    with pytest.raises(RuntimeError, match="differs"):
        check_greedy_output(target, inputs, response, 4, [])


def save_fixture(tmp_path):
    target, draft = make_real_models(layers=2, loops=3)
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target.save_pretrained(target_path)
    draft.save_pretrained(draft_path)
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(32)}, unk_token="0"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="0").save_pretrained(
        target_path
    )
    dataset = tmp_path / "toy.jsonl"
    dataset.write_text('{"prompt": "5 9 2"}\n{"prompt": "7 4"}\n', encoding="utf-8")
    return leagle.build_parser().parse_args(
        [
            "--verifier-model",
            str(target_path),
            "--draft-model",
            str(draft_path),
            "--datasets-root",
            str(dataset),
            "--output-dir",
            str(tmp_path / "results"),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "12",
            "--raw-prompt-mode",
            "raw",
            "--check-greedy",
            "--no-progress",
        ]
    )


def test_saved_checkpoint_to_dataset_results_and_provenance(tmp_path):
    args = save_fixture(tmp_path)
    leagle.run(args, entrypoint=Path("scripts/evaluate/leagle_offline_eval.py"))
    rows = json.loads((args.output_dir / "summary.json").read_text())
    assert rows[0]["num_requests"] == 2
    assert 1 <= rows[0]["acceptance_length"] <= 8
    assert rows[0]["probability_diagnostics_available"] is False
    assert rows[0]["num_proposals"] > 0
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["draft_config"]["num_loops"] == 3
    assert metadata["arguments"]["check_greedy"] is True
    assert metadata["arguments"]["draft_attn_impl"] == "sdpa"
    assert (args.output_dir / "eval_command.txt").is_file()
    assert (args.output_dir / "speculators.patch").is_file()
    samples = [
        json.loads(line)
        for line in (args.output_dir / "artifacts/toy.jsonl").read_text().splitlines()
    ]
    assert len(samples) == 2
    assert all("accepted_draft_lengths" in sample for sample in samples)


def test_worker_preserves_budget_sdpa_and_audit_and_shard_statistics(tmp_path):
    args = save_fixture(tmp_path)
    args.spec_tokens = 3
    args.ascend_devices = "0,1"
    command = parallel.worker_command(
        args,
        entrypoint=Path("scripts/evaluate/leagle_offline_eval.py"),
        dataset_path=args.datasets_root,
        shard_index=1,
        num_shards=2,
        output_dir=tmp_path / "worker",
        progress_path=tmp_path / "worker.count",
    )
    worker_args = leagle.build_parser().parse_args(command[2:])
    assert worker_args.spec_tokens == 3
    assert worker_args.target_attn_impl == worker_args.draft_attn_impl == "sdpa"
    assert worker_args.check_greedy
    assert worker_args.worker_shard_index == 1
    stats_a = EvalStats(probability_diagnostics_available=False)
    stats_b = EvalStats(probability_diagnostics_available=False)
    for stats, accepted in [(stats_a, [7, 7]), (stats_b, [0])]:
        stats.add_response(
            SimpleNamespace(
                num_output_tokens=10,
                proposal_lengths=[7] * len(accepted),
                accepted_draft_lengths=accepted,
            )
        )
    row = aggregate_rows(
        "toy", [summary_row("toy", 1, stats) for stats in [stats_a, stats_b]]
    )
    assert row["acceptance_length"] == pytest.approx(1 + 14 / 3)


def test_reject_sampling_invalid_budget_and_conflicting_vocab_mapping(tmp_path):
    args = save_fixture(tmp_path)
    args.temperature = 0.6
    with pytest.raises(ValueError, match="temperature=0"):
        leagle.validate_args(args)
    args.temperature = 0
    args.spec_tokens = 0
    with pytest.raises(ValueError, match="spec-tokens"):
        leagle.validate_args(args)
    args.spec_tokens = 7
    args.d2t_path = tmp_path / "d2t.pt"
    with pytest.raises(ValueError, match="together"):
        leagle.validate_args(args)
    args.t2d_path = tmp_path / "t2d.pt"
    mask = torch.zeros(32, dtype=torch.bool)
    mask[1::2] = True
    torch.save(mask, args.t2d_path)
    torch.save(torch.arange(1, 32, 2) - torch.arange(16), args.d2t_path)
    with pytest.raises(ValueError, match="Cannot replace"):
        leagle.load_models(args)
