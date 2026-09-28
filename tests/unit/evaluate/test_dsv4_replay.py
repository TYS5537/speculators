"""CPU replay tests: real packets/cache files and the unchanged decoding loop."""

# ruff: noqa: INP001 -- Existing evaluate tests are not a package.

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from speculators_dsv4.block_protocol import GREEDY_REQUEST_KEY
from speculators_dsv4.replay import DSV4GreedyReplayTarget
from speculators_dsv4.replay_cache import GreedyTraceCache, trace_identity
from speculators_eval.profiling import EvaluationProfiler
from speculators_eval.reporting import EvalStats, aggregate_rows, summary_row


def _evaluator():
    path = (
        Path(__file__).resolve().parents[3] / "scripts/evaluate/dspark_offline_eval.py"
    )
    spec = importlib.util.spec_from_file_location("replay_evaluator_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.torch = torch
    return module


@pytest.fixture
def target(tmp_path):
    value = DSV4GreedyReplayTarget.__new__(DSV4GreedyReplayTarget)
    value.layer_ids = [1, 11]
    value.packet_layer_ids = [1, 11, 43]
    value.manifest = {"checkpoint_signature": "test", "runtime_quantization": None}
    value.vocab_size, value.hidden_size = 11, 2
    value.max_model_len = 512
    value.model_name = "target"
    value.device = torch.device("cpu")
    value.timeout = 120
    value.http_transfer = None
    value.hidden_states_path = tmp_path / "hs"
    value.hidden_states_path.mkdir()
    value.keep_hidden_states = False
    value.verification_mode = "block"
    value.profiler = EvaluationProfiler(enabled=True, device="cpu")
    value.num_target_requests = 0
    value.trace_cache = GreedyTraceCache(tmp_path / "traces")
    value.replay_tag = "test"
    value.live_audit = False
    value.audit_remaining = 0
    value.trace = None
    value.calls = []
    value.corrupt_block = False

    def create(**kwargs):
        value.calls.append(kwargs)
        prefix = kwargs["prompt"]
        request_id = kwargs["extra_body"]["request_id"]
        params = kwargs["extra_body"]["kv_transfer_params"]
        if GREEDY_REQUEST_KEY in params:
            assert kwargs["temperature"] == 0
            assert kwargs["extra_body"]["ignore_eos"] is True
            ids = [(prefix[-1] + i + 1) % 11 for i in range(kwargs["max_tokens"])]
            transfer = {GREEDY_REQUEST_KEY: {"version": 1}}
            choice = SimpleNamespace(prompt_token_ids=prefix, token_ids=ids)
        else:
            options = params["dsv4_block_verify"]
            start, hidden_start = options["logits_start"], options["hidden_start"]
            ids = (torch.tensor(prefix[start:]) + 1) % 11
            if value.corrupt_block:
                ids[0] = (ids[0] + 1) % 11
            hidden = torch.tensor(prefix).cumsum(0).to(torch.bfloat16)
            hidden = hidden[:, None, None].expand(-1, 3, 2).contiguous()
            path = value.hidden_states_path / f"cmpl-{request_id}-0.safetensors"
            save_file(
                {
                    "token_ids": torch.tensor(prefix, dtype=torch.int64),
                    "verification_metadata": torch.tensor(
                        [2, len(prefix), start, hidden_start]
                    ),
                    "layer_ids": torch.tensor(value.packet_layer_ids),
                    "greedy_token_ids": ids,
                    "hidden_states": hidden[hidden_start:].clone(),
                    "server_timings": torch.zeros(3, dtype=torch.float64),
                },
                path,
            )
            transfer = {"hidden_states_path": str(path), "dsv4_block_verify_version": 2}
            choice = SimpleNamespace(prompt_token_ids=prefix)
        return SimpleNamespace(
            id=f"cmpl-{request_id}",
            model=value.model_name,
            choices=[choice],
            kv_transfer_params=transfer,
        )

    value.client = SimpleNamespace(completions=SimpleNamespace(create=create))
    value.configure_evaluation(
        temperature=0, requires_target_logits=False, block_output="auto"
    )
    return value


def test_trace_generation_chunking_reuse_and_identity(target):
    prompt = torch.tensor([[1, 2]])
    target.prepare_sample(prompt, 260, [])
    assert len(target.calls) == 4  # One native generation + three <=128-row extracts.
    assert target.trace["hidden_states"].shape == (261, 2, 2)
    assert not list(target.hidden_states_path.glob("*.safetensors"))
    target.prepare_sample(prompt, 260, [])
    assert len(target.calls) == 4
    target.replay_tag = "changed-runtime"
    target.prepare_sample(prompt, 260, [])
    assert len(target.calls) == 8
    target.manifest["runtime_quantization"] = "changed"
    target.prepare_sample(prompt, 260, [])
    assert len(target.calls) == 12


def test_failed_native_argmax_check_never_publishes_cache(target):
    target.corrupt_block = True
    with pytest.raises(ValueError, match="disagree"):
        target.prepare_sample(torch.tensor([[1, 2]]), 10, [])
    assert target.trace is None
    assert not target.trace_cache.directory.exists()


def test_cache_checksum_failure_is_fatal_without_regeneration(target):
    prompt = torch.tensor([[1, 2]])
    target.prepare_sample(prompt, 6, [])
    identity = trace_identity(target, [1, 2], 6, [], "test")
    path = target.trace_cache.path(identity) / "trace.safetensors"
    path.write_bytes(path.read_bytes() + b"corruption")
    calls = len(target.calls)
    with pytest.raises(ValueError, match="checksum"):
        target.prepare_sample(prompt, 6, [])
    assert len(target.calls) == calls


@pytest.mark.parametrize(
    ("prompt", "budget"),
    [([], 4), ([[1], [2]], 4), ([[1, 2]], 0), ([[1, 2]], 511)],
)
def test_invalid_input_fails_before_cache_or_rpc(target, prompt, budget):
    with pytest.raises(ValueError):
        target.prepare_sample(torch.tensor(prompt, dtype=torch.long), budget, [])
    assert not target.calls
    assert not target.trace_cache.directory.exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_backend": "hf"},
        {"temperature": 1.0},
        {"dsv4_kv_reuse": True},
        {"dsv4_block_output": "full"},
        {"measure_base_speedup": True},
        {"dsv4_replay_audit_samples": -1},
    ],
)
def test_invalid_replay_cli_fails_before_loading_models(overrides):
    module = _evaluator()
    options = {
        "target_backend": "dsv4-vllm",
        "dsv4_verification_mode": "replay",
        "temperature": 0.0,
        "dsv4_replay_audit_samples": 0,
        **overrides,
    }
    with pytest.raises(ValueError, match="Replay"):
        module.run(SimpleNamespace(**options))


@pytest.mark.parametrize(
    ("temperature", "requires", "mode"),
    [(1, False, "auto"), (0, True, "auto"), (0, False, "full")],
)
def test_replay_rejects_unsupported_metrics(target, temperature, requires, mode):
    with pytest.raises(ValueError, match="requires"):
        target.configure_evaluation(
            temperature=temperature, requires_target_logits=requires, block_output=mode
        )


@pytest.mark.parametrize(
    ("budget", "stops"), [(1, []), (2, []), (7, []), (19, []), (19, [3]), (19, [6])]
)
@pytest.mark.parametrize("wrong_offset", [0, 1, 2, 5])
def test_replay_matches_live_acceptance_and_never_leaks_future(
    target, budget, stops, wrong_offset
):
    module = _evaluator()
    prompt = torch.tensor([[1, 2]])
    target.prepare_sample(prompt, budget, stops)
    prepare_calls = len(target.calls)

    def init_context(*, initial_output, initial_token):
        del initial_token
        assert initial_output.hidden_states[1].shape[1] == 2
        return {"hidden_length": 2}

    def propose(*, context, output_ids, position_ids, start, stop_token_ids):
        del position_ids, stop_token_ids
        assert context["hidden_length"] == start
        anchor = int(output_ids[0, start])
        ids = [(anchor + i + 1) % 11 for i in range(4)]
        if wrong_offset < len(ids):
            ids[wrong_offset] = (ids[wrong_offset] + 3) % 11
        tensor = torch.tensor([ids])
        return module.DraftProposal(
            4,
            torch.tensor([[anchor, *ids]]),
            torch.nn.functional.one_hot(tensor, 11).float(),
        )

    def update(context, verification):
        # The current anchor and accepted candidates only; no rejected/future HS.
        length = verification.accepted_draft_tokens + 1
        states = verification.target_output.hidden_states[1][:, :length]
        context["hidden_length"] += states.shape[1]

    def generate():
        return module.generate_decoding_sample(
            target_model=target,
            input_ids=prompt,
            max_new_tokens=budget,
            max_proposal_tokens=4,
            temperature=0,
            stop_token_ids=stops,
            init_context=init_context,
            propose=propose,
            update=update,
        )

    replayed = generate()
    assert len(target.calls) == prepare_calls
    assert replayed.output_ids[0].tolist() == target.trace["tokens"].tolist()
    target.audit_remaining = 1
    target.audit(generate, replayed)
    assert target.audit_remaining == 0
    assert not target.live_audit
    stats = EvalStats(probability_diagnostics_available=False)
    stats.add_response(replayed)
    row = summary_row("test", 1, stats)
    merged = aggregate_rows("test", [row, row])
    assert merged["num_proposals"] == row["num_proposals"] * 2
    assert merged["acceptance_length"] == row["acceptance_length"]
    assert json.loads(merged["position_accept_prob_means"]) == []
    assert not merged["probability_diagnostics_available"]


def test_replay_rejects_wrong_committed_prefix(target):
    target.prepare_sample(torch.tensor([[1, 2]]), 10, [])
    cache = target.new_cache()
    cache.tokens = [1, 0]
    proposal = SimpleNamespace(
        draft_token_count=1, verify_input_ids=torch.tensor([[3, 4]])
    )
    with pytest.raises(ValueError, match="committed"):
        target.verify_proposal(
            proposal=proposal, cache=cache, start=2, stop_token_ids=[]
        )


def test_live_audit_detects_mismatch_and_restores_mode(target):
    target.audit_remaining = 1
    result = SimpleNamespace(
        output_ids=torch.tensor([[1]]), proposal_lengths=[2], accepted_draft_lengths=[1]
    )
    live = SimpleNamespace(**vars(result))
    live.accepted_draft_lengths = [0]
    with pytest.raises(ValueError, match="audit diverged"):
        target.audit(lambda: live, result)
    assert not target.live_audit
    assert target.audit_remaining == 1
