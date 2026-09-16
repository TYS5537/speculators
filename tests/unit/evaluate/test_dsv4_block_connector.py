"""CPU tests for the server connector: real tensors/files, isolated vLLM stubs."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4 import offline as offline_backend

ROOT = Path(__file__).resolve().parents[3]


class _ConnectorBase:
    def __init__(self, vllm_config, role, kv_cache_config):
        self._kv_transfer_config = vllm_config.kv_transfer_config
        self._connector_metadata = None

    def has_connector_metadata(self):
        return self._connector_metadata is not None

    def bind_connector_metadata(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _Metadata:
    pass


class _SupportsHMA:
    pass


class _TPGroup:
    def __init__(self):
        self.flags = []
        self.remote_error = False

    def all_reduce(self, tensor):
        self.flags.append(tensor.detach().clone())
        if self.remote_error:
            return torch.ones_like(tensor)
        return tensor


@pytest.fixture
def connector_module(monkeypatch):
    transfer = ModuleType("vllm.distributed.kv_transfer")
    transfer.has_kv_transfer_group = lambda: False
    transfer.get_kv_transfer_group = lambda: None
    base = ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
    base.KVConnectorBase_V1 = _ConnectorBase
    base.KVConnectorMetadata = _Metadata
    base.SupportsHMA = _SupportsHMA
    parallel = ModuleType("vllm.distributed.parallel_state")
    group = _TPGroup()
    parallel.get_tensor_model_parallel_rank = lambda: 0
    parallel.get_tp_group = lambda: group
    for name, module in (
        (transfer.__name__, transfer),
        (base.__name__, base),
        (parallel.__name__, parallel),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    path = ROOT / "src/speculators_dsv4/block_connector.py"
    spec = importlib.util.spec_from_file_location("dsv4_block_connector_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.test_group = group
    return module


def _config(directory, *, max_num_seqs=1, **extra):
    settings = {"shared_storage_path": str(directory), **extra}
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        kv_transfer_config=SimpleNamespace(get_from_extra_config=settings.get),
        speculative_config=SimpleNamespace(
            method="extract_hidden_states",
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 11, 43]),
            ),
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=5, hidden_size=4),
        ),
    )


def _request(*, tokens=None, logits_start=2, hidden_start=1):
    return SimpleNamespace(
        req_id="cmpl-test-0",
        prompt_token_ids=list(tokens if tokens is not None else [1, 2, 3, 4]),
        num_computed_tokens=0,
        prompt_embeds=None,
        mm_features=[],
        lora_request=None,
        sampling_params=SimpleNamespace(
            max_tokens=1,
            n=1,
            logprobs=None,
            prompt_logprobs=None,
            extra_args={
                "kv_transfer_params": {
                    "dsv4_block_verify": {
                        "version": 1,
                        "logits_start": logits_start,
                        "hidden_start": hidden_start,
                    },
                },
            },
        ),
    )


def _schedule(request):
    return SimpleNamespace(
        scheduled_new_reqs=[request],
        num_scheduled_tokens={request.req_id: len(request.prompt_token_ids)},
    )


@pytest.fixture
def ready(connector_module, tmp_path):
    connector = connector_module.DSV4BlockVerifyConnector(
        _config(tmp_path),
        "worker",
        None,
    )
    request = _request()
    metadata = connector.build_connector_meta(_schedule(request))
    connector.bind_connector_metadata(metadata)
    normalized = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    auxiliary = [normalized + 40 * (index + 1) for index in range(3)]
    # Poison padded rows so accidental inclusion cannot silently pass.
    normalized[4:] = torch.nan
    for value in auxiliary:
        value[4:] = torch.nan
    calls = []
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0], [4.0, 2.0, 0.0, 1.0, 3.0]])

    def compute_logits(hidden):
        calls.append(hidden.clone())
        return logits.clone()

    return SimpleNamespace(
        module=connector_module,
        connector=connector,
        request=request,
        metadata=metadata,
        model=SimpleNamespace(compute_logits=compute_logits),
        input_ids=torch.tensor([1, 2, 3, 4, 0, 0]),
        positions=torch.arange(6),
        output=(normalized, auxiliary),
        calls=calls,
        logits=logits,
        path=Path(metadata.request.filename),
    )


def _capture(ready):
    ready.connector.capture(ready.model, ready.input_ids, ready.positions, ready.output)


def test_scheduler_copies_prefix_and_returns_versioned_handle(ready):
    ready.request.prompt_token_ids[0] = 4
    assert ready.metadata.request.token_ids == [1, 2, 3, 4]
    request = SimpleNamespace(request_id=ready.request.req_id)
    delayed, handle = ready.connector.request_finished_all_groups(request, ([], []))
    assert delayed is False
    assert handle == {
        "hidden_states_path": str(ready.path),
        "dsv4_block_verify_version": 1,
    }
    assert ready.connector.request_finished(request, []) == (False, None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_token_ids", []),
        ("prompt_token_ids", None),
        ("prompt_token_ids", [True, 2, 3, 4]),
        ("prompt_token_ids", [5, 2, 3, 4]),
        ("prompt_token_ids", [-1, 2, 3, 4]),
        ("prompt_token_ids", [1.0, 2, 3, 4]),
        ("num_computed_tokens", 1),
        ("prompt_embeds", torch.zeros(4, 4)),
        ("mm_features", [object()]),
        ("lora_request", object()),
        ("req_id", "../outside"),
        ("sampling_params", None),
    ],
)
def test_scheduler_rejects_non_full_token_prefix(ready, field, value):
    request = _request()
    scheduled = _schedule(request)
    setattr(request, field, value)
    if field == "req_id":
        scheduled.num_scheduled_tokens = {value: 4}
    with pytest.raises(ValueError):
        ready.connector.build_connector_meta(scheduled)


@pytest.mark.parametrize(("field", "value"), [("max_tokens", 2), ("n", 2)])
def test_scheduler_requires_one_output_and_one_sequence(ready, field, value):
    request = _request()
    setattr(request.sampling_params, field, value)
    with pytest.raises(ValueError, match="max_tokens=1 and n=1"):
        ready.connector.build_connector_meta(_schedule(request))


@pytest.mark.parametrize("field", ["logprobs", "prompt_logprobs"])
def test_scheduler_rejects_http_probability_materialization(ready, field):
    request = _request()
    setattr(request.sampling_params, field, 5)
    with pytest.raises(ValueError, match="not HTTP logprobs"):
        ready.connector.build_connector_meta(_schedule(request))


def test_scheduler_rejects_duplicate_inflight_id(ready):
    with pytest.raises(ValueError, match="Duplicate in-flight"):
        ready.connector.build_connector_meta(_schedule(_request()))


@pytest.mark.parametrize(
    "params",
    [
        None,
        [],
        {"other": {}},
        {"dsv4_block_verify": {"version": 2, "logits_start": 2, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": -1, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": 4, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": True, "hidden_start": 1}},
        {"dsv4_block_verify": {"version": 1, "logits_start": 2, "hidden_start": 5}},
        {"dsv4_block_verify": {"version": 1, "logits_start": 2}},
        {"dsv4_block_verify": {}, "hidden_states_path": "outside.safetensors"},
    ],
)
def test_scheduler_rejects_missing_or_bad_protocol(ready, params):
    request = _request()
    request.sampling_params.extra_args = {"kv_transfer_params": params}
    with pytest.raises(ValueError):
        ready.connector.build_connector_meta(_schedule(request))


def test_scheduler_rejects_chunks_cached_work_and_multiple_requests(ready):
    request = _request()
    scheduled = _schedule(request)
    scheduled.num_scheduled_tokens[request.req_id] = 3
    with pytest.raises(ValueError, match="unchunked"):
        ready.connector.build_connector_meta(scheduled)
    scheduled.scheduled_new_reqs = []
    with pytest.raises(ValueError, match="fresh full-prefix"):
        ready.connector.build_connector_meta(scheduled)
    scheduled.scheduled_new_reqs = [request, _request()]
    with pytest.raises(ValueError, match="only one"):
        ready.connector.build_connector_meta(scheduled)
    scheduled.scheduled_new_reqs = []
    scheduled.num_scheduled_tokens = {}
    assert ready.connector.build_connector_meta(scheduled).request is None


def test_connector_requires_single_sequence_and_storage(connector_module, tmp_path):
    with pytest.raises(ValueError, match="max-num-seqs"):
        connector_module.DSV4BlockVerifyConnector(
            _config(tmp_path, max_num_seqs=2),
            "worker",
            None,
        )
    with pytest.raises(ValueError, match="storage directory"):
        connector_module.DSV4BlockVerifyConnector(
            _config(tmp_path, shared_storage_path=""),
            "worker",
            None,
        )


def test_connector_never_loads_or_defers_kv(ready):
    assert ready.connector.get_num_new_matched_tokens(None, 0) == (0, False)
    ready.connector.update_state_after_alloc(None, None, 0)
    with pytest.raises(ValueError, match="external KV"):
        ready.connector.update_state_after_alloc(None, None, 1)
    ready.connector.start_load_kv(None)
    ready.connector.wait_for_layer_load("anything")
    ready.connector.save_kv_layer(None, None, None)
    ready.connector.wait_for_save()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_connector_requires_positive_integer_row_limit(
    connector_module, tmp_path, limit
):
    with pytest.raises(ValueError, match="max_verify_rows"):
        connector_module.DSV4BlockVerifyConnector(
            _config(tmp_path, max_verify_rows=limit),
            "worker",
            None,
        )


def test_scheduler_caps_head_rows_not_prefix_length(connector_module, tmp_path):
    connector = connector_module.DSV4BlockVerifyConnector(
        _config(tmp_path, max_verify_rows=2),
        "worker",
        None,
    )
    request = _request(tokens=[1] * 129, logits_start=126, hidden_start=0)
    with pytest.raises(ValueError, match="exceeds 2"):
        connector.build_connector_meta(_schedule(request))
    request = _request(tokens=[1] * 129, logits_start=127, hidden_start=0)
    assert (
        connector.build_connector_meta(_schedule(request)).request.logits_start == 127
    )


def test_capture_projects_only_normalized_suffix_and_real_packet_roundtrip(ready):
    _capture(ready)
    assert len(ready.calls) == 1
    torch.testing.assert_close(ready.calls[0], ready.output[0][2:4])
    assert not torch.equal(ready.calls[0], ready.output[1][-1][2:4])
    packet = load_file(str(ready.path))
    assert packet["verification_metadata"].tolist() == [1, 4, 2, 1]
    assert packet["token_ids"].tolist() == [1, 2, 3, 4]
    assert packet["layer_ids"].tolist() == [1, 11, 43]
    assert packet["logprobs"].dtype == torch.float32
    assert packet["hidden_states"].dtype == torch.bfloat16
    assert packet["hidden_states"].shape == (3, 3, 4)
    torch.testing.assert_close(
        packet["logprobs"],
        torch.log_softmax(ready.logits.float(), dim=-1),
    )
    torch.testing.assert_close(
        packet["hidden_states"],
        torch.stack([value[1:4] for value in ready.output[1]], 1),
    )
    assert list(ready.path.parent.iterdir()) == [ready.path]
    assert len(ready.module.test_group.flags) == 1
    assert ready.module.test_group.flags[0].dtype == torch.int32
    assert ready.module.test_group.flags[0].item() == 0


def test_nonroot_also_enters_native_head_before_rank_gate(ready, monkeypatch):
    events = []

    def head(hidden):
        events.append("head")
        torch.testing.assert_close(hidden, ready.output[0][2:4])

    def rank():
        events.append("rank")
        return 1

    ready.model.compute_logits = head
    monkeypatch.setattr(ready.module, "get_tensor_model_parallel_rank", rank)
    _capture(ready)
    assert events.index("head") < events.index("rank")
    assert not ready.path.exists()
    assert len(ready.module.test_group.flags) == 1


def test_capture_skips_warmup_and_rejects_wrong_metadata(ready):
    ready.connector.bind_connector_metadata(None)
    _capture(ready)
    ready.connector.bind_connector_metadata(ready.module.BlockMetadata())
    _capture(ready)
    assert not ready.calls
    ready.connector.bind_connector_metadata(object())
    with pytest.raises(ValueError, match="metadata"):
        _capture(ready)


@pytest.mark.parametrize("field", ["input_ids", "positions"])
def test_capture_rejects_prefix_position_mismatch_before_head(ready, field):
    value = getattr(ready, field).clone()
    value[0] += 1
    setattr(ready, field, value)
    with pytest.raises(ValueError, match="tokens/positions"):
        _capture(ready)
    assert not ready.calls


@pytest.mark.parametrize("case", ["dtype", "width", "length", "layer_count"])
def test_capture_rejects_bad_hidden_shape_or_dtype(ready, case):
    normalized, auxiliary = ready.output
    auxiliary = list(auxiliary)
    if case == "dtype":
        auxiliary[0] = auxiliary[0].float()
    elif case == "width":
        auxiliary[0] = auxiliary[0][:, :3]
    elif case == "length":
        auxiliary[0] = auxiliary[0][:3]
    else:
        auxiliary.pop()
    ready.output = (normalized, auxiliary)
    with pytest.raises(ValueError, match="shape/dtype"):
        _capture(ready)
    assert not ready.calls


@pytest.mark.parametrize("case", ["nan_logits", "infinite_logits", "nan_hidden"])
def test_capture_rejects_nonfinite_export(ready, case):
    if case == "nan_logits":
        ready.logits[0, 0] = torch.nan
    elif case == "infinite_logits":
        ready.logits[0] = -torch.inf
    else:
        ready.output[1][0][1, 0] = torch.nan
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "nonfinite" in str(exc_info.value.__cause__)
    assert ready.module.test_group.flags[0].item() == 1
    assert not ready.path.exists()


@pytest.mark.parametrize("value", [None, torch.zeros(1, 5), torch.zeros(2, 4)])
def test_capture_rejects_missing_or_truncated_vocab(ready, value):
    ready.model.compute_logits = lambda _: value
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "logits shape" in str(exc_info.value.__cause__)
    assert not ready.path.exists()


def test_atomic_packet_publish_and_collision_never_overwrite(ready, monkeypatch):
    original_save = ready.module.save_file
    checked = []

    def save(tensors, temporary):
        assert not ready.path.exists()
        assert Path(temporary).parent == ready.path.parent
        original_save(tensors, temporary)
        checked.append(temporary)

    monkeypatch.setattr(ready.module, "save_file", save)
    _capture(ready)
    assert len(checked) == 1
    original_bytes = ready.path.read_bytes()
    with pytest.raises(FileExistsError):
        ready.module._save_packet({"x": torch.ones(1)}, str(ready.path))
    assert ready.path.read_bytes() == original_bytes
    assert list(ready.path.parent.iterdir()) == [ready.path]


def test_failed_packet_write_removes_only_temporary_file(ready, monkeypatch):
    def fail_save(tensors, filename):
        raise OSError("simulated full disk")

    monkeypatch.setattr(ready.module, "save_file", fail_save)
    with pytest.raises(OSError, match="full disk"):
        ready.module._save_packet({"x": torch.ones(1)}, str(ready.path))
    assert not ready.path.exists()
    assert list(ready.path.parent.iterdir()) == []


def test_atomic_publish_collision_race_preserves_other_artifact(ready, monkeypatch):
    original_save = ready.module.save_file

    def race_save(tensors, temporary):
        original_save(tensors, temporary)
        original_save({"other_request": torch.tensor([7])}, str(ready.path))

    monkeypatch.setattr(ready.module, "save_file", race_save)
    with pytest.raises(FileExistsError):
        ready.module._save_packet({"new_request": torch.tensor([8])}, str(ready.path))
    assert load_file(str(ready.path))["other_request"].tolist() == [7]
    assert list(ready.path.parent.iterdir()) == [ready.path]


def test_root_io_failure_is_collectively_reported(ready, monkeypatch):
    def fail_write(*args):
        raise OSError("simulated full disk")

    monkeypatch.setattr(ready.module, "_save_packet", fail_write)
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert isinstance(exc_info.value.__cause__, OSError)
    assert ready.module.test_group.flags[0].item() == 1


def test_nonroot_receives_root_failure_after_native_head(ready, monkeypatch):
    monkeypatch.setattr(ready.module, "get_tensor_model_parallel_rank", lambda: 1)
    ready.module.test_group.remote_error = True
    with pytest.raises(RuntimeError, match="packet export failed") as exc_info:
        _capture(ready)
    assert exc_info.value.__cause__ is None
    assert len(ready.calls) == 1
    assert ready.module.test_group.flags[0].item() == 0
    assert not ready.path.exists()


def test_export_dispatch_requires_correct_connector(ready, monkeypatch):
    ready.module.export_block(None, None, None, None)
    monkeypatch.setattr(ready.module, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(ready.module, "get_kv_transfer_group", object)
    with pytest.raises(ValueError, match="dedicated connector"):
        ready.module.export_block(None, None, None, None)
    monkeypatch.setattr(ready.module, "get_kv_transfer_group", lambda: ready.connector)
    ready.module.export_block(
        ready.model, ready.input_ids, ready.positions, ready.output
    )
    assert ready.path.exists()


class _TinyDraft(torch.nn.Module):
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


def test_real_connector_packet_to_offline_target_prefill_and_suffix(
    connector_module,
    tmp_path,
    monkeypatch,
):
    """Only HTTP/model execution is faked; both transport ends use real code."""
    connector = connector_module.DSV4BlockVerifyConnector(
        _config(tmp_path),
        "worker",
        None,
    )
    model_path = tmp_path / "checkpoint"
    draft = _TinyDraft(model_path)
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
    monkeypatch.setattr(offline_backend, "ensure_manifest", lambda *args: None)
    recorded = []

    def create(**kwargs):
        prefix = kwargs["prompt"]
        body = kwargs["extra_body"]
        request_id = body["request_id"]
        assert "logprobs" not in kwargs
        assert "prompt_logprobs" not in body
        request = _request(tokens=prefix)
        request.req_id = f"cmpl-{request_id}-0"
        request.sampling_params.max_tokens = kwargs["max_tokens"]
        request.sampling_params.n = kwargs["n"]
        request.sampling_params.extra_args = {
            "kv_transfer_params": body["kv_transfer_params"],
        }
        metadata = connector.build_connector_meta(_schedule(request))
        connector.bind_connector_metadata(metadata)
        normalized = torch.arange(len(prefix), dtype=torch.float32).view(-1, 1)
        normalized = normalized.expand(-1, 4).to(torch.bfloat16)
        auxiliary = [normalized + 10 * (index + 1) for index in range(3)]

        def head(hidden):
            recorded.append(
                (list(prefix), hidden.shape[0], metadata.request.hidden_start)
            )
            weights = torch.arange(5, dtype=torch.float32).view(1, -1)
            return hidden[:, :1].float() * weights

        connector.capture(
            SimpleNamespace(compute_logits=head),
            torch.tensor(prefix),
            torch.arange(len(prefix)),
            (normalized, auxiliary),
        )
        _, handle = connector.request_finished(
            SimpleNamespace(request_id=request.req_id),
            [],
        )
        return SimpleNamespace(
            id=f"cmpl-{request_id}",
            model="test-target",
            choices=[SimpleNamespace(prompt_token_ids=list(prefix))],
            kv_transfer_params=handle,
        )

    target = offline_backend.DSV4OfflineTarget(
        draft,
        report,
        hidden_states_path=tmp_path,
        client=SimpleNamespace(completions=SimpleNamespace(create=create)),
        model_name="test-target",
        max_model_len=64,
        verification_mode="block",
    )
    cache = target.new_cache()
    prefill = target(
        input_ids=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        past_key_values=cache,
        output_hidden_states=True,
    )
    suffix = target(
        input_ids=torch.tensor([[4, 0]]),
        position_ids=torch.tensor([[3, 4]]),
        past_key_values=cache,
        output_hidden_states=True,
    )
    assert recorded == [([1, 2, 3], 1, 0), ([1, 2, 3, 4, 0], 2, 3)]
    assert target.num_target_requests == 2
    assert cache.tokens == [1, 2, 3, 4, 0]
    assert prefill.logits.shape == (1, 1, 5)
    assert suffix.logits.shape == (1, 2, 5)
    torch.testing.assert_close(
        prefill.logits[0, 0],
        torch.log_softmax(torch.arange(5).float() * 2, 0),
    )
    torch.testing.assert_close(
        suffix.logits[0],
        torch.log_softmax(torch.tensor([[3.0], [4.0]]) * torch.arange(5), -1),
    )
    assert prefill.hidden_states[1].shape == (1, 3, 4)
    torch.testing.assert_close(
        suffix.hidden_states[1],
        torch.tensor([[[13.0] * 4, [14.0] * 4]], dtype=torch.bfloat16),
    )
    assert list(tmp_path.iterdir()) == []
