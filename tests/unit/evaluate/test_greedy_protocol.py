"""The AR and block drafters share acceptance, EOS and budget accounting."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark import DSparkDraftModel, DSparkSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators_eval.leagle_decoding import generate_greedy
from speculators_eval.reporting import EvalStats, aggregate_rows, summary_row

from .test_dspark_offline_eval import _load_module
from .test_leagle_offline_eval import (
    ControlledDraft,
    CountingTarget,
    make_real_models,
    save_fixture,
)


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 9, 17])
@pytest.mark.parametrize("stop", [None, 2, 7, 10])
@pytest.mark.parametrize("reject_at", [None, 0, 2])
def test_block_and_ar_round_accounting_match(budget, stop, reject_at):
    module = _load_module()
    module.DynamicCache = DynamicCache
    stops = [] if stop is None else [stop]
    inputs = torch.tensor([[0, 1]])

    def propose(*, output_ids, start, **kwargs):
        anchor = output_ids[:, start : start + 1]
        proposed = (anchor + torch.arange(1, 8)) % 64
        if reject_at is not None:
            proposed[:, reject_at] = 60
        return module.DraftProposal(
            7,
            torch.cat([anchor, proposed], dim=1),
            torch.nn.functional.one_hot(proposed, 64).float(),
        )

    block = module.generate_decoding_sample(
        target_model=CountingTarget().eval(),
        input_ids=inputs,
        max_new_tokens=budget,
        max_proposal_tokens=7,
        temperature=0,
        stop_token_ids=stops,
        init_context=lambda **_: None,
        propose=propose,
        update=lambda *_: None,
    )
    ar = generate_greedy(
        CountingTarget().eval(),
        ControlledDraft(reject_at).eval(),
        inputs,
        max_new_tokens=budget,
        stop_token_ids=stops,
    )
    torch.testing.assert_close(block.output_ids, ar.output_ids)
    assert block.proposal_lengths == ar.proposal_lengths
    assert block.accepted_draft_lengths == ar.accepted_draft_lengths


def test_dspark_proposal_budget_caps_slots_and_rejects_unsupported_lengths():
    module = _load_module()
    draft = SimpleNamespace(
        block_size=8,
        use_draft_vocab=False,
        config=SimpleNamespace(sample_from_anchor=True),
    )
    args = SimpleNamespace(spec_tokens=3, temperature=0)
    runner = module.DSparkOfflineRunner(torch.nn.Linear(1, 1), draft, None, args)
    ids, probabilities = runner._sample_dspark_tokens(
        torch.randn(1, 8, 32), torch.randn(1, 8, 16), torch.tensor([5])
    )
    assert len(ids) == probabilities.shape[1] == 3
    args.spec_tokens = 9
    with pytest.raises(ValueError, match="spec-tokens"):
        module.DSparkOfflineRunner(torch.nn.Linear(1, 1), draft, None, args)


def test_per_round_rates_include_short_tails_and_aggregate_counts():
    left, right = (
        EvalStats(probability_diagnostics_available=False),
        EvalStats(probability_diagnostics_available=False),
    )
    left.add_response(
        SimpleNamespace(
            num_output_tokens=5, proposal_lengths=[3, 0], accepted_draft_lengths=[2, 0]
        )
    )
    right.add_response(
        SimpleNamespace(
            num_output_tokens=2, proposal_lengths=[1], accepted_draft_lengths=[1]
        )
    )
    result = aggregate_rows("toy", [summary_row("toy", 1, s) for s in (left, right)])
    assert json.loads(result["position_accept_rates"]) == [1, 1, 0]
    assert json.loads(result["position_accept_rates_per_round"]) == pytest.approx(
        [2 / 3, 1 / 3, 0]
    )
    assert result["acceptance_length"] == 2


def test_dspark_checkpoint_to_jsonl_greedy_audit_and_provenance(tmp_path, monkeypatch):
    args = save_fixture(tmp_path)
    target, _ = make_real_models()
    decoder = copy.deepcopy(target.config)
    decoder.num_hidden_layers = 1
    decoder.layer_types = ["full_attention"]
    draft = DSparkDraftModel(
        DSparkSpeculatorConfig(
            transformer_layer_config=decoder,
            draft_vocab_size=32,
            block_size=8,
            aux_hidden_state_layer_ids=[1, 2],
            mask_token_id=0,
            sample_from_anchor=True,
            speculators_config=SpeculatorsConfig(
                algorithm="dspark",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=8)],
                default_proposal_method="greedy",
                verifier=VerifierConfig(
                    name_or_path=None, architectures=["Qwen3ForCausalLM"]
                ),
            ),
        )
    )
    draft.save_pretrained(args.draft_model)
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dspark_offline_eval.py",
            "--verifier-model",
            args.verifier_model,
            "--draft-model",
            args.draft_model,
            "--datasets-root",
            str(args.datasets_root),
            "--output-dir",
            str(args.output_dir),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--draft-attn-impl",
            "sdpa",
            "--target-attn-impl",
            "sdpa",
            "--spec-tokens",
            "3",
            "--max-new-tokens",
            "12",
            "--raw-prompt-mode",
            "raw",
            "--check-greedy",
            "--no-progress",
        ],
    )
    module.main()
    rows = json.loads((args.output_dir / "summary.json").read_text())
    assert rows[0]["num_requests"] == 2
    assert 1 <= rows[0]["acceptance_length"] <= 4
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["algorithm"] == "dspark"
    assert metadata["spec_tokens"] == 3
    assert (args.output_dir / "drafter_checkpoint_sha256.txt").is_file()
