"""CPU adapter tests: mock transport, real tensors and existing acceptance math."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4 import offline as backend

ROOT = Path(__file__).resolve().parents[3]


def _load_evaluator():
    path = ROOT / "scripts/evaluate/dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dsv4_offline_eval_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.torch = torch
    return module


class TinyDraft(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.target_layer_ids = [1, 11]
        self.config = SimpleNamespace(
            target_hidden_state_format=HS_FORMAT,
            speculators_config=SimpleNamespace(
                verifier=SimpleNamespace(name_or_path=str(model_path)),
            ),
        )


def _hidden_for(tokens):
    """Each position/layer has a different exact BF16 integer for slicing checks."""
    positions = torch.arange(len(tokens), dtype=torch.float32).view(-1, 1, 1)
    slots = torch.arange(3, dtype=torch.float32).view(1, -1, 1)
    return (positions * 10 + slots).expand(-1, -1, 4).to(torch.bfloat16)


@pytest.fixture
def target_fixture(tmp_path, monkeypatch):
    model_path = tmp_path / "target"
    model_path.mkdir()
    report = {
        "model_path": str(model_path),
        "checkpoint_signature": "test-checkpoint",
        "config": {
            "model_type": "deepseek_v4",
            "hidden_size": 4,
            "num_hidden_layers": 43,
            "vocab_size": 5,
            "eos_token_id": 4,
        },
    }
    monkeypatch.setattr(backend, "ensure_manifest", lambda *args, **kwargs: None)
    draft = TinyDraft(model_path)
    target = backend.DSV4OfflineTarget(
        draft_model=draft,
        report=report,
        hidden_states_path=tmp_path / "hs",
        client=object(),
        model_name="served-target",
        max_model_len=64,
    )
    requests = []
    logprobs = [math.log(value) for value in [0.05, 0.1, 0.15, 0.3, 0.4]]

    def request(prefix, need_hidden):
        requests.append((list(prefix), need_hidden))
        return logprobs, _hidden_for(prefix) if need_hidden else None

    monkeypatch.setattr(target, "_request", request)
    return SimpleNamespace(
        target=target,
        draft=draft,
        requests=requests,
        logprobs=logprobs,
        request=request,
        report=report,
    )


def _forward(target, cache, tokens, *, hidden=True):
    start = cache.get_seq_length()
    return target(
        input_ids=torch.tensor([tokens], dtype=torch.long),
        position_ids=torch.arange(start, start + len(tokens)).unsqueeze(0),
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=hidden,
    )


def test_prefill_requests_only_last_distribution_but_keeps_all_prompt_hs(
    target_fixture,
):
    target = target_fixture.target
    cache = target.new_cache()
    output = _forward(target, cache, [2, 0, 3])

    assert target_fixture.requests == [([2, 0, 3], True)]
    assert cache.tokens == [2, 0, 3]
    assert output.logits.shape == (1, 1, 5)
    torch.testing.assert_close(
        output.logits.float(),
        torch.tensor(target_fixture.logprobs).view(1, 1, 5),
    )
    for slot, layer in enumerate(target_fixture.draft.target_layer_ids):
        torch.testing.assert_close(
            output.hidden_states[layer], _hidden_for([2, 0, 3])[:, slot].unsqueeze(0)
        )


def test_verify_requests_every_prefix_and_returns_only_new_suffix(target_fixture):
    target = target_fixture.target
    cache = target.new_cache()
    _forward(target, cache, [2, 0])
    target_fixture.requests.clear()

    output = _forward(target, cache, [1, 3, 4])

    assert target_fixture.requests == [
        ([2, 0, 1], False),
        ([2, 0, 1, 3], False),
        ([2, 0, 1, 3, 4], True),
    ]
    assert cache.tokens == [2, 0, 1, 3, 4]
    assert output.logits.shape == (1, 3, 5)
    full_hidden = _hidden_for(cache.tokens)
    for slot, layer in enumerate(target_fixture.draft.target_layer_ids):
        assert output.hidden_states[layer].shape == (1, 3, 4)
        torch.testing.assert_close(
            output.hidden_states[layer], full_hidden[2:, slot].unsqueeze(0)
        )


def test_crop_excludes_rejected_tokens_from_next_remote_prefix(target_fixture):
    target = target_fixture.target
    cache = target.new_cache()
    _forward(target, cache, [2, 0])
    _forward(target, cache, [1, 3, 4])
    # Anchor 1 and candidate 3 accepted; candidate 4 rejected, next anchor is 0.
    cache.crop(4)
    target_fixture.requests.clear()

    output = _forward(target, cache, [0, 2])

    assert target_fixture.requests == [
        ([2, 0, 1, 3, 0], False),
        ([2, 0, 1, 3, 0, 2], True),
    ]
    assert cache.tokens == [2, 0, 1, 3, 0, 2]
    torch.testing.assert_close(
        output.hidden_states[1], _hidden_for(cache.tokens)[4:, 0].unsqueeze(0)
    )


def test_position_mismatch_fails_before_request_and_does_not_mutate_cache(
    target_fixture,
):
    target = target_fixture.target
    cache = target.new_cache()
    _forward(target, cache, [2, 0])
    target_fixture.requests.clear()

    with pytest.raises(ValueError):
        target(
            input_ids=torch.tensor([[1, 3]]),
            position_ids=torch.tensor([[1, 2]]),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
    assert target_fixture.requests == []
    assert cache.tokens == [2, 0]


def test_new_target_cache_uses_backend_factory_and_preserves_hf_fallback():
    evaluator = _load_evaluator()
    service_cache = object()
    hf_cache = object()
    evaluator.DynamicCache = lambda: hf_cache

    assert (
        evaluator._new_target_cache(SimpleNamespace(new_cache=lambda: service_cache))
        is service_cache
    )
    assert evaluator._new_target_cache(object()) is hf_cache


def test_ascend_worker_command_preserves_remote_target_configuration(
    tmp_path, monkeypatch
):
    evaluator = _load_evaluator()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dspark_offline_eval.py",
            "--verifier-model",
            str(tmp_path / "target"),
            "--draft-model",
            str(tmp_path / "draft"),
            "--datasets-root",
            str(tmp_path / "datasets"),
            "--target-backend",
            "dsv4-vllm",
            "--vllm-endpoint",
            "http://target-host:8001/v1",
            "--hidden-states-path",
            str(tmp_path / "hs"),
            "--served-model-name",
            "served-target",
            "--dsv4-max-model-len",
            "8192",
            "--dsv4-verification-mode",
            "block",
            "--target-request-timeout",
            "321",
            "--keep-target-hs",
        ],
    )
    args = evaluator.parse_args()
    command = evaluator._worker_command(
        args,
        dataset_path=tmp_path / "datasets/sample.jsonl",
        shard_index=2,
        num_shards=4,
        output_dir=tmp_path / "output",
    )
    expected = {
        "--target-backend": "dsv4-vllm",
        "--vllm-endpoint": "http://target-host:8001/v1",
        "--hidden-states-path": str(tmp_path / "hs"),
        "--served-model-name": "served-target",
        "--dsv4-max-model-len": "8192",
        "--dsv4-verification-mode": "block",
        "--target-request-timeout": "321.0",
        "--worker-shard-index": "2",
        "--worker-num-shards": "4",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value
    assert "--keep-target-hs" in command
    assert "--measure-base-speedup" not in command


def test_service_logprobs_preserve_temperature_distribution_and_greedy():
    evaluator = _load_evaluator()
    logits = torch.tensor([[[1.0, 2.0, -1.0, 0.0]]])
    server_logprobs = logits.log_softmax(-1)

    for temperature in (0.0, 0.5, 1.0, 1.7):
        torch.testing.assert_close(
            evaluator.logits_to_probs(server_logprobs, temperature),
            evaluator.logits_to_probs(logits, temperature),
        )


def test_greedy_verify_accepts_matching_prefix_and_rejects_first_mismatch(
    target_fixture, monkeypatch
):
    evaluator = _load_evaluator()
    target = target_fixture.target
    probabilities = [0.1, 0.6, 0.1, 0.1, 0.1]

    def request(prefix, need_hidden):
        return (
            [math.log(value) for value in probabilities],
            _hidden_for(prefix) if need_hidden else None,
        )

    monkeypatch.setattr(target, "_request", request)
    cache = target.new_cache()
    _forward(target, cache, [3, 0])
    proposal = evaluator.DraftProposal(
        draft_token_count=2,
        verify_input_ids=torch.tensor([[2, 1, 4]]),
        draft_probs=torch.nn.functional.one_hot(
            torch.tensor([[1, 4]]), num_classes=5
        ).float(),
    )
    verification = evaluator.verify_draft_tokens(
        target_model=target,
        proposal=proposal,
        position_ids=torch.arange(8).unsqueeze(0),
        start=2,
        past_key_values_target=cache,
        temperature=0.0,
        max_proposal_tokens=2,
        current_token_ids=torch.tensor([[2]]),
    )

    assert verification.accepted_draft_tokens == 1
    assert verification.next_token.item() == 1
    assert verification.accept_probs.tolist() == [[1.0, 0.0]]
    cache.crop(4)
    assert cache.tokens == [3, 0, 2, 1]


def test_temperature_one_uses_probability_ratio_and_residual_distribution(
    target_fixture, monkeypatch
):
    evaluator = _load_evaluator()
    target = target_fixture.target
    probabilities = [0.1, 0.2, 0.4, 0.2, 0.1]
    draft_probs = torch.tensor([[[0.2, 0.4, 0.2, 0.1, 0.1]]])

    def request(prefix, need_hidden):
        return (
            [math.log(value) for value in probabilities],
            _hidden_for(prefix) if need_hidden else None,
        )

    monkeypatch.setattr(target, "_request", request)
    monkeypatch.setattr(torch, "rand_like", lambda values: torch.full_like(values, 0.9))
    residuals = []

    def choose_highest(probs):
        residuals.append(probs.clone())
        return probs.argmax(dim=-1)

    monkeypatch.setattr(evaluator, "sample_from_probs", choose_highest)
    cache = target.new_cache()
    _forward(target, cache, [3, 0])
    verification = evaluator.verify_draft_tokens(
        target_model=target,
        proposal=evaluator.DraftProposal(
            draft_token_count=1,
            verify_input_ids=torch.tensor([[2, 1]]),
            draft_probs=draft_probs,
        ),
        position_ids=torch.arange(6).unsqueeze(0),
        start=2,
        past_key_values_target=cache,
        temperature=1.0,
        max_proposal_tokens=1,
        current_token_ids=torch.tensor([[2]]),
    )

    assert verification.accepted_draft_tokens == 0
    assert verification.accept_probs.item() == pytest.approx(0.5)
    assert verification.support_accept_rates.item() == pytest.approx(0.7)
    assert verification.next_token.item() == 2
    torch.testing.assert_close(
        residuals[0],
        torch.tensor([[0.0, 0.0, 2 / 3, 1 / 3, 0.0]]),
        atol=1e-6,
        rtol=1e-6,
    )


def test_decoding_loop_keeps_anchor_out_of_context_until_verification(
    target_fixture, monkeypatch
):
    evaluator = _load_evaluator()
    target = target_fixture.target
    target.max_model_len = 5  # Prompt + output fits without a full extra draft block.
    cache = target.new_cache()
    monkeypatch.setattr(target, "new_cache", lambda: cache)
    contexts = []

    def init_context(*, initial_output, initial_token):
        assert initial_token.tolist() == [[4]]
        context = SimpleNamespace(hidden=initial_output.hidden_states[1])
        contexts.append(context)
        return context

    def propose(*, context, output_ids, position_ids, start, stop_token_ids):
        assert start == 2
        assert context.hidden.shape == (1, start, 4)
        assert cache.get_seq_length() == start
        assert output_ids[:, start].item() == 4
        return evaluator.DraftProposal(
            draft_token_count=2,
            verify_input_ids=torch.tensor([[4, 4, 4]]),
            draft_probs=torch.nn.functional.one_hot(
                torch.tensor([[4, 4]]), num_classes=5
            ).float(),
        )

    def update(context, verification):
        pytest.fail("The final verification must not prepare another draft round")

    result = evaluator.generate_decoding_sample(
        target_model=target,
        input_ids=torch.tensor([[2, 0]]),
        max_new_tokens=3,
        max_proposal_tokens=2,
        temperature=0.0,
        stop_token_ids=None,
        init_context=init_context,
        propose=propose,
        update=update,
    )

    assert result.output_ids.tolist() == [[2, 0, 4, 4, 4]]
    assert result.num_output_tokens == 3
    assert result.proposal_lengths == [1]
    assert result.accepted_draft_lengths == [1]
    assert result.accept_prob_lists == [[1.0]]
    assert target_fixture.requests == [
        ([2, 0], True),
        ([2, 0, 4], False),
        ([2, 0, 4, 4], True),
    ]
    assert cache.tokens == [2, 0, 4, 4]
    assert contexts[0].hidden.shape == (1, 2, 4)


def test_request_failure_does_not_commit_candidate_tokens(target_fixture, monkeypatch):
    target = target_fixture.target
    cache = target.new_cache()
    _forward(target, cache, [2, 0])

    def request(prefix, need_hidden):
        raise RuntimeError("simulated transport failure")

    monkeypatch.setattr(target, "_request", request)
    with pytest.raises(RuntimeError, match="simulated transport failure"):
        _forward(target, cache, [1, 3])
    assert cache.tokens == [2, 0]


@pytest.mark.parametrize("max_proposal_tokens", [0, 3, 128])
@pytest.mark.parametrize(("prompt_length", "max_new_tokens"), [(60, 4), (63, 1)])
def test_context_budget_allows_exact_output_limit(
    target_fixture, prompt_length, max_new_tokens, max_proposal_tokens
):
    target = target_fixture.target
    target.validate_request_budget(prompt_length, max_new_tokens, max_proposal_tokens)


def test_context_budget_rejects_one_token_over_output_limit(target_fixture):
    target = target_fixture.target
    with pytest.raises(ValueError, match="65 target positions"):
        target.validate_request_budget(61, 4, 3)


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_context_budget_rejects_nonpositive_generation_length(
    target_fixture, max_new_tokens
):
    with pytest.raises(ValueError, match="max_new_tokens must be > 0"):
        target_fixture.target.validate_request_budget(4, max_new_tokens, 3)


@pytest.fixture
def request_fixture(target_fixture, monkeypatch):
    target = target_fixture.target
    request_id = "0123456789abcdef0123456789abcdef"
    prefix = [2, 0, 3]
    target.hidden_states_path.mkdir()
    path = target.hidden_states_path / f"cmpl-{request_id}-0.safetensors"
    path.touch()
    top = {
        f"token_id:{token}": value
        for token, value in enumerate(target_fixture.logprobs)
    }
    response = SimpleNamespace(
        id=f"cmpl-{request_id}",
        model=target.model_name,
        choices=[
            SimpleNamespace(
                prompt_token_ids=prefix,
                logprobs=SimpleNamespace(top_logprobs=[top]),
            )
        ],
        kv_transfer_params={"hidden_states_path": str(path)},
    )
    calls = []
    reads = []
    deleted = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    def get_generated(handle):
        reads.append(handle)
        return {"token_ids": torch.tensor(prefix), "hidden_states": _hidden_for(prefix)}

    def delete(handle):
        deleted.append(handle)
        Path(handle).unlink()

    fake_connectors = ModuleType("hs_connectors")
    fake_connectors.FileTransfer = lambda directory: SimpleNamespace(
        get_generated=get_generated, delete=delete
    )
    fake_transfer = ModuleType("hs_connectors.transfer")
    fake_transfer.wait_for_lock = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "hs_connectors", fake_connectors)
    monkeypatch.setitem(sys.modules, "hs_connectors.transfer", fake_transfer)
    monkeypatch.setattr(backend, "uuid4", lambda: SimpleNamespace(hex=request_id))
    target.client = SimpleNamespace(completions=SimpleNamespace(create=create))
    return SimpleNamespace(
        target=target,
        prefix=prefix,
        response=response,
        path=path,
        calls=calls,
        reads=reads,
        deleted=deleted,
        transfer=fake_transfer,
        request_id=request_id,
    )


def test_real_request_uses_full_vocab_neutral_parameters_and_deletes_own_file(
    request_fixture,
):
    case = request_fixture
    probabilities, hidden = backend.DSV4OfflineTarget._request(
        case.target, case.prefix, need_hidden=True
    )

    call = case.calls[0]
    assert call["prompt"] == case.prefix
    assert call["logprobs"] == 5
    assert call["temperature"] == 1.0
    assert call["top_p"] == 1.0
    assert call["max_tokens"] == 1
    assert call["frequency_penalty"] == 0.0
    assert call["presence_penalty"] == 0.0
    assert call["extra_body"]["top_k"] in (-1, 0)
    assert call["extra_body"]["min_p"] == 0.0
    assert call["extra_body"]["repetition_penalty"] == 1.0
    assert call["extra_body"]["return_tokens_as_token_ids"] is True
    assert call["extra_body"]["return_token_ids"] is True
    assert call["extra_body"]["add_special_tokens"] is False
    assert call["extra_body"]["ignore_eos"] is True
    assert case.target.num_target_requests == 1
    assert len(probabilities) == 5
    torch.testing.assert_close(hidden, _hidden_for(case.prefix))
    assert case.reads == [str(case.path)]
    assert case.deleted == [str(case.path)]
    assert not case.path.exists()


def test_intermediate_request_does_not_read_hs_but_cleans_owned_file(request_fixture):
    case = request_fixture
    probabilities, hidden = backend.DSV4OfflineTarget._request(
        case.target, case.prefix, need_hidden=False
    )

    assert len(probabilities) == 5
    assert hidden is None
    assert case.reads == []
    assert case.deleted == [str(case.path)]


def test_request_rejects_truncated_logprobs_and_cleans_owned_file(request_fixture):
    case = request_fixture
    case.response.choices[0].logprobs.top_logprobs = [{"token_id:0": 0.0}]
    with pytest.raises(ValueError, match="full-vocabulary"):
        backend.DSV4OfflineTarget._request(case.target, case.prefix, need_hidden=True)
    assert case.reads == []
    assert case.deleted == [str(case.path)]


def test_mismatched_response_id_cannot_delete_a_file(request_fixture):
    case = request_fixture
    case.response.id = "cmpl-someone-elses-request"
    with pytest.raises(ValueError, match="request ID"):
        backend.DSV4OfflineTarget._request(case.target, case.prefix, need_hidden=True)
    assert case.reads == []
    assert case.deleted == []
    assert case.path.exists()


def test_lock_timeout_leaves_writer_files_intact(request_fixture, monkeypatch):
    case = request_fixture
    lock = Path(str(case.path) + ".lock")
    lock.touch()

    def wait(*args, **kwargs):
        raise TimeoutError("simulated writer timeout")

    monkeypatch.setattr(case.transfer, "wait_for_lock", wait)
    with pytest.raises(TimeoutError, match="simulated writer timeout"):
        backend.DSV4OfflineTarget._request(case.target, case.prefix, need_hidden=True)
    assert case.reads == []
    assert case.deleted == []
    assert case.path.exists()
    assert lock.exists()


def _native_logprobs(prefix):
    # A deterministic causal target with position- and context-sensitive output.
    logits = torch.tensor([0.25, -0.75, 0.5, 0.0, 1.25])
    logits[(sum(prefix) + len(prefix)) % len(logits)] += 1.5
    return logits.log_softmax(-1)


@pytest.fixture
def block_fixture(target_fixture):
    target = target_fixture.target
    target.verification_mode = "block"
    target.hidden_states_path.mkdir()
    case = SimpleNamespace(
        target=target,
        draft=target_fixture.draft,
        report=target_fixture.report,
        calls=[],
        paths=[],
        packet_hook=lambda packet: None,
        response_hook=lambda response: None,
        transport_failure=False,
    )

    def create(**kwargs):
        case.calls.append(kwargs)
        if case.transport_failure:
            raise RuntimeError("simulated block transport failure")
        prefix = kwargs["prompt"]
        request_id = kwargs["extra_body"]["request_id"]
        options = kwargs["extra_body"]["kv_transfer_params"]["dsv4_block_verify"]
        logits_start, hidden_start = options["logits_start"], options["hidden_start"]
        path = target.hidden_states_path / f"cmpl-{request_id}-0.safetensors"
        packet = {
            "token_ids": torch.tensor(prefix, dtype=torch.int64),
            "verification_metadata": torch.tensor(
                [1, len(prefix), logits_start, hidden_start], dtype=torch.int64
            ),
            "layer_ids": torch.tensor([1, 11, 43], dtype=torch.int64),
            "logprobs": torch.stack(
                [
                    _native_logprobs(prefix[: row + 1])
                    for row in range(logits_start, len(prefix))
                ]
            ),
            "hidden_states": _hidden_for(prefix)[hidden_start:].contiguous(),
        }
        case.packet_hook(packet)
        save_file(packet, str(path))
        case.paths.append(path)
        response = SimpleNamespace(
            id=f"cmpl-{request_id}",
            model=target.model_name,
            choices=[SimpleNamespace(prompt_token_ids=prefix)],
            kv_transfer_params={
                "dsv4_block_verify_version": 1,
                "hidden_states_path": str(path),
            },
        )
        case.response_hook(response)
        return response

    target.client = SimpleNamespace(completions=SimpleNamespace(create=create))
    return case


def test_block_prefill_exports_last_logprob_and_complete_prompt_hs(block_fixture):
    case = block_fixture
    cache = case.target.new_cache()
    output = _forward(case.target, cache, [2, 0, 3])

    assert len(case.calls) == case.target.num_target_requests == 1
    call = case.calls[0]
    assert call["prompt"] == [2, 0, 3]
    assert call.get("logprobs") is None
    assert call["temperature"] == call["top_p"] == 1.0
    assert call["max_tokens"] == call["n"] == 1
    assert call["frequency_penalty"] == call["presence_penalty"] == 0.0
    assert call["extra_body"]["ignore_eos"] is True
    assert call["extra_body"]["add_special_tokens"] is False
    assert call["extra_body"]["return_token_ids"] is True
    assert call["extra_body"]["top_k"] == call["extra_body"]["min_p"] == 0
    assert call["extra_body"]["repetition_penalty"] == 1.0
    assert call["extra_body"]["kv_transfer_params"] == {
        "dsv4_block_verify": {"version": 1, "logits_start": 2, "hidden_start": 0}
    }
    assert cache.tokens == [2, 0, 3]
    torch.testing.assert_close(
        output.logits, _native_logprobs(cache.tokens).view(1, 1, 5)
    )
    for slot, layer in enumerate(case.draft.target_layer_ids):
        torch.testing.assert_close(
            output.hidden_states[layer], _hidden_for(cache.tokens)[:, slot].unsqueeze(0)
        )
    assert not case.paths[0].exists()


def test_block_verifies_entire_suffix_once_and_returns_only_suffix_hs(block_fixture):
    case = block_fixture
    cache = case.target.new_cache()
    _forward(case.target, cache, [2, 0])
    output = _forward(case.target, cache, [1, 3, 4])

    assert len(case.calls) == case.target.num_target_requests == 2
    assert case.calls[-1]["prompt"] == [2, 0, 1, 3, 4]
    assert case.calls[-1]["extra_body"]["kv_transfer_params"]["dsv4_block_verify"] == {
        "version": 1,
        "logits_start": 2,
        "hidden_start": 2,
    }
    assert output.logits.shape == (1, 3, 5)
    for slot, layer in enumerate(case.draft.target_layer_ids):
        torch.testing.assert_close(
            output.hidden_states[layer],
            _hidden_for(cache.tokens)[2:, slot].unsqueeze(0),
        )


def test_block_crop_excludes_rejected_tokens_from_next_prefix(block_fixture):
    case = block_fixture
    cache = case.target.new_cache()
    _forward(case.target, cache, [2, 0])
    _forward(case.target, cache, [1, 3, 4])
    cache.crop(4)
    output = _forward(case.target, cache, [0, 2])
    assert case.calls[-1]["prompt"] == [2, 0, 1, 3, 0, 2]
    assert output.logits.shape == (1, 2, 5)
    torch.testing.assert_close(
        output.hidden_states[1], _hidden_for(cache.tokens)[4:, 0].unsqueeze(0)
    )


@pytest.mark.parametrize("hidden", [False, True])
def test_block_matches_reference_logits_hs_after_prefill_and_crop(
    block_fixture, monkeypatch, hidden
):
    case = block_fixture
    reference = backend.DSV4OfflineTarget(
        case.draft,
        case.report,
        hidden_states_path=case.target.hidden_states_path,
        client=object(),
        model_name="served-target",
        max_model_len=64,
    )
    monkeypatch.setattr(
        reference,
        "_request",
        lambda prefix, need_hidden: (
            _native_logprobs(prefix).tolist(),
            _hidden_for(prefix) if need_hidden else None,
        ),
    )
    caches = [reference.new_cache(), case.target.new_cache()]
    for tokens in ([2, 0], [1, 3, 4], [0, 2]):
        outputs = [
            _forward(target, cache, tokens, hidden=hidden)
            for target, cache in zip([reference, case.target], caches, strict=True)
        ]
        torch.testing.assert_close(outputs[0].logits, outputs[1].logits)
        if hidden:
            for layer in case.draft.target_layer_ids:
                torch.testing.assert_close(
                    outputs[0].hidden_states[layer], outputs[1].hidden_states[layer]
                )
        else:
            assert outputs[0].hidden_states is outputs[1].hidden_states is None
            assert case.calls[-1]["extra_body"]["kv_transfer_params"][
                "dsv4_block_verify"
            ]["hidden_start"] == len(case.calls[-1]["prompt"])
        assert caches[0].tokens == caches[1].tokens
        if tokens == [1, 3, 4]:
            for cache in caches:
                cache.crop(4)


@pytest.mark.parametrize("keep", [False, True])
def test_block_keep_flag_controls_only_own_packet_cleanup(block_fixture, keep):
    case = block_fixture
    case.target.keep_hidden_states = keep
    unrelated = case.target.hidden_states_path / "other-request.safetensors"
    unrelated.touch()
    _forward(case.target, case.target.new_cache(), [2, 0])
    assert case.paths[-1].exists() is keep
    assert unrelated.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_ids", torch.tensor([2, 1], dtype=torch.int64)),
        ("token_ids", torch.tensor([2, 0], dtype=torch.int32)),
        ("verification_metadata", torch.tensor([2, 2, 1, 0], dtype=torch.int64)),
        ("verification_metadata", torch.tensor([1, 2, 0, 0], dtype=torch.int64)),
        ("verification_metadata", torch.tensor([1, 2, 1, 1], dtype=torch.int64)),
        ("verification_metadata", torch.tensor([1, 2, 1, 0], dtype=torch.int32)),
        ("layer_ids", torch.tensor([11, 1, 43], dtype=torch.int64)),
        ("layer_ids", torch.tensor([1, 11, 42], dtype=torch.int64)),
        ("layer_ids", torch.tensor([1, 11, 43], dtype=torch.int32)),
        ("logprobs", torch.zeros(1, 5, dtype=torch.float32)),
        ("logprobs", torch.full((1, 5), -math.inf, dtype=torch.float32)),
        ("logprobs", torch.full((1, 5), math.inf, dtype=torch.float32)),
        ("logprobs", torch.full((1, 5), math.nan, dtype=torch.float32)),
        ("logprobs", torch.zeros(1, 4, dtype=torch.float32)),
        ("logprobs", torch.ones(1, 5, dtype=torch.bfloat16)),
        ("hidden_states", torch.zeros(2, 3, 4, dtype=torch.float32)),
        ("hidden_states", torch.zeros(1, 3, 4, dtype=torch.bfloat16)),
        ("hidden_states", torch.full((2, 3, 4), math.nan, dtype=torch.bfloat16)),
    ],
)
def test_invalid_block_packet_never_commits_cache_and_cleans_owned_packet(
    block_fixture, field, value
):
    case = block_fixture
    cache = case.target.new_cache()
    case.packet_hook = lambda packet: packet.__setitem__(field, value)
    with pytest.raises(ValueError):
        _forward(case.target, cache, [2, 0])
    assert cache.tokens == []
    assert case.target.num_target_requests == 1
    assert not case.paths[-1].exists()


@pytest.mark.parametrize("extra_field", [False, True])
def test_block_packet_rejects_missing_or_extra_schema_fields(
    block_fixture, extra_field
):
    case = block_fixture

    def invalidate(packet):
        if extra_field:
            packet["unknown"] = torch.ones(1)
        else:
            del packet["layer_ids"]

    case.packet_hook = invalidate
    cache = case.target.new_cache()
    with pytest.raises(ValueError, match="fields"):
        _forward(case.target, cache, [2, 0])
    assert cache.tokens == []
    assert not case.paths[-1].exists()


def test_block_packet_accepts_zero_probability_tokens(block_fixture):
    case = block_fixture
    case.packet_hook = lambda packet: packet.__setitem__(
        "logprobs", torch.tensor([[-math.inf, 0.0, -math.inf, -math.inf, -math.inf]])
    )
    output = _forward(case.target, case.target.new_cache(), [2, 0])
    assert output.logits.argmax(-1).item() == 1


@pytest.mark.parametrize("version", [None, True, "1", 2])
def test_missing_or_invalid_block_confirmation_fails_without_fallback(
    block_fixture, version
):
    case = block_fixture
    case.response_hook = lambda response: response.kv_transfer_params.__setitem__(
        "dsv4_block_verify_version", version
    )
    cache = case.target.new_cache()
    with pytest.raises(ValueError, match="protocol version"):
        _forward(case.target, cache, [2, 0])
    assert cache.tokens == []
    assert len(case.calls) == 1
    # Without the atomic block contract, this could still be a legacy writer's file.
    assert case.paths[-1].exists()


def test_block_response_prefix_mismatch_fails_before_cache_commit(block_fixture):
    case = block_fixture
    case.response_hook = lambda response: setattr(
        response.choices[0], "prompt_token_ids", [0]
    )
    cache = case.target.new_cache()
    with pytest.raises(ValueError, match="changed/truncated"):
        _forward(case.target, cache, [2, 0])
    assert cache.tokens == []
    assert not case.paths[-1].exists()


@pytest.mark.parametrize("wrong_id", [False, True])
def test_block_unowned_handle_or_response_never_deletes_a_file(block_fixture, wrong_id):
    case = block_fixture
    unrelated = case.target.hidden_states_path / "someone-else.safetensors"
    unrelated.touch()

    def invalidate(response):
        if wrong_id:
            response.id = "cmpl-someone-elses-request"
        else:
            response.kv_transfer_params["hidden_states_path"] = str(unrelated)

    case.response_hook = invalidate
    cache = case.target.new_cache()
    with pytest.raises(ValueError):
        _forward(case.target, cache, [2, 0])
    assert cache.tokens == []
    assert case.paths[-1].exists()
    assert unrelated.exists()


def test_block_rejects_legacy_writer_lock_without_deleting_active_file(block_fixture):
    case = block_fixture
    case.response_hook = lambda response: Path(
        response.kv_transfer_params["hidden_states_path"] + ".lock"
    ).touch()
    cache = case.target.new_cache()
    with pytest.raises(ValueError, match="writer lock"):
        _forward(case.target, cache, [2, 0])
    assert cache.tokens == []
    assert case.paths[-1].exists()
    assert Path(str(case.paths[-1]) + ".lock").exists()


def test_block_transport_failure_does_not_commit_candidates(block_fixture):
    case = block_fixture
    cache = case.target.new_cache()
    _forward(case.target, cache, [2, 0])
    case.transport_failure = True
    with pytest.raises(RuntimeError, match="block transport failure"):
        _forward(case.target, cache, [1, 3])
    assert cache.tokens == [2, 0]
    assert len(case.calls) == case.target.num_target_requests == 2


@pytest.mark.parametrize("mode", ["reference", "block"])
def test_backend_metadata_identifies_transport_but_not_online_speedup(tmp_path, mode):
    evaluator = _load_evaluator()
    args = SimpleNamespace(
        output_dir=tmp_path,
        hidden_states_path=tmp_path / "hs",
        dsv4_verification_mode=mode,
        dsv4_max_model_len=4096,
        temperature=1.0,
    )
    evaluator._write_backend_metadata(
        args, {"model_path": "target", "checkpoint_signature": "test"}
    )
    metadata = json.loads((tmp_path / "eval_backend.json").read_text(encoding="utf-8"))
    assert metadata["verification_mode"] == mode
    assert metadata["online_speedup_benchmark"] is False
    assert metadata["verification"] == (
        "full-prefix-block-recompute"
        if mode == "block"
        else "full-prefix-per-position-recompute"
    )
    assert metadata["position_accept_rates"] == "accepted_prefix_count / proposed_count"


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("accepted_eos", [False, True])
def test_block_reference_acceptance_and_eos_statistics_match(
    block_fixture, monkeypatch, temperature, accepted_eos
):
    case = block_fixture
    evaluator = _load_evaluator()
    probabilities = torch.tensor(
        [0.1, 0.1, 0.1, 0.1, 0.6] if accepted_eos else [0.1, 0.2, 0.4, 0.2, 0.1]
    )
    logprobs = probabilities.log()
    case.packet_hook = lambda packet: packet.__setitem__(
        "logprobs", logprobs.repeat(packet["logprobs"].shape[0], 1)
    )
    reference = backend.DSV4OfflineTarget(
        case.draft,
        case.report,
        hidden_states_path=case.target.hidden_states_path,
        client=object(),
        model_name="served-target",
        max_model_len=64,
    )
    monkeypatch.setattr(
        reference,
        "_request",
        lambda prefix, need_hidden: (
            logprobs.tolist(),
            _hidden_for(prefix) if need_hidden else None,
        ),
    )
    monkeypatch.setattr(torch, "rand_like", lambda values: torch.full_like(values, 0.9))
    monkeypatch.setattr(evaluator, "sample_from_probs", lambda probs: probs.argmax(-1))
    candidates = [4, 4] if accepted_eos else [2, 1]
    draft_probs = probabilities.repeat(1, 2, 1)
    if not accepted_eos:
        draft_probs[:, 1] = torch.tensor([0.1, 0.5, 0.1, 0.2, 0.1])
    proposal = evaluator.DraftProposal(
        draft_token_count=2,
        verify_input_ids=torch.tensor([[3, *candidates]]),
        draft_probs=draft_probs,
    )
    results = []
    for target in (reference, case.target):
        cache = target.new_cache()
        _forward(target, cache, [2, 0])
        results.append(
            evaluator.verify_draft_tokens(
                target_model=target,
                proposal=proposal,
                position_ids=torch.arange(8).unsqueeze(0),
                start=2,
                past_key_values_target=cache,
                temperature=temperature,
                max_proposal_tokens=2,
                current_token_ids=torch.tensor([[3]]),
                stop_token_ids=[4],
            )
        )
    for field in (
        "target_probs",
        "accept_prefix_mask",
        "accept_probs",
        "support_accept_rates",
        "next_token",
        "committed_tokens",
    ):
        torch.testing.assert_close(
            getattr(results[0], field), getattr(results[1], field)
        )
    for result in results:
        assert result.accepted_draft_tokens == 1
        assert result.terminated_by_stop_token is accepted_eos
        assert result.effective_proposal_length == (1 if accepted_eos else 2)
        assert result.accept_probs.shape[-1] == (1 if accepted_eos else 2)
    assert len(case.calls) == 2


def test_block_initial_eos_ends_before_any_proposal(block_fixture, monkeypatch):
    case = block_fixture
    evaluator = _load_evaluator()
    case.packet_hook = lambda packet: packet.__setitem__(
        "logprobs", torch.tensor([[-math.inf, -math.inf, -math.inf, -math.inf, 0.0]])
    )

    def unexpected(**kwargs):
        pytest.fail("Initial EOS must not initialize draft context or propose tokens")

    result = evaluator.generate_decoding_sample(
        target_model=case.target,
        input_ids=torch.tensor([[2, 0]]),
        max_new_tokens=3,
        max_proposal_tokens=2,
        temperature=0.0,
        stop_token_ids=[4],
        init_context=unexpected,
        propose=unexpected,
        update=unexpected,
    )
    assert result.output_ids.tolist() == [[2, 0, 4]]
    assert result.proposal_lengths == result.accepted_draft_lengths == []
    assert result.accept_prob_lists == result.support_accept_rate_lists == []
    assert case.target.num_target_requests == 1


def test_block_last_token_uses_target_only_at_exact_context_limit(
    block_fixture, monkeypatch
):
    case = block_fixture
    evaluator = _load_evaluator()
    case.target.max_model_len = 4
    cache = case.target.new_cache()
    monkeypatch.setattr(case.target, "new_cache", lambda: cache)
    packet_shapes = []
    case.packet_hook = lambda packet: packet_shapes.append(packet["logprobs"].shape)

    def unexpected(*args, **kwargs):
        pytest.fail("A target-only final token must not propose or update a draft")

    result = evaluator.generate_decoding_sample(
        target_model=case.target,
        input_ids=torch.tensor([[2, 0]]),
        max_new_tokens=2,
        max_proposal_tokens=2,
        temperature=0.0,
        stop_token_ids=None,
        init_context=lambda **kwargs: None,
        propose=unexpected,
        update=unexpected,
    )

    assert result.output_ids.tolist() == [[2, 0, 4, 4]]
    assert result.num_output_tokens == 2
    assert result.proposal_lengths == result.accepted_draft_lengths == [0]
    assert result.accept_prob_lists == result.support_accept_rate_lists == [[]]
    assert cache.tokens == [2, 0, 4]
    assert len(case.calls) == case.target.num_target_requests == 2
    assert [call["prompt"] for call in case.calls] == [[2, 0], [2, 0, 4]]
    assert case.calls[-1]["extra_body"]["kv_transfer_params"]["dsv4_block_verify"] == {
        "version": 1,
        "logits_start": 2,
        "hidden_start": 2,
    }
    assert packet_shapes == [(1, 5), (1, 5)]


def test_block_corrupt_packet_does_not_commit_existing_prefix(block_fixture):
    case = block_fixture
    cache = case.target.new_cache()
    _forward(case.target, cache, [2, 0])
    case.response_hook = lambda response: Path(
        response.kv_transfer_params["hidden_states_path"]
    ).write_bytes(b"not a safetensors packet")
    with pytest.raises(Exception, match="header"):
        _forward(case.target, cache, [1, 3])
    assert cache.tokens == [2, 0]
    assert not case.paths[-1].exists()


def test_block_bad_suffix_packet_does_not_commit_candidates(block_fixture):
    case = block_fixture
    cache = case.target.new_cache()
    _forward(case.target, cache, [2, 0])
    case.packet_hook = lambda packet: packet.__setitem__(
        "verification_metadata", torch.tensor([1, 4, 1, 2], dtype=torch.int64)
    )
    with pytest.raises(ValueError, match="verification_metadata"):
        _forward(case.target, cache, [1, 3])
    assert cache.tokens == [2, 0]
    assert not case.paths[-1].exists()
