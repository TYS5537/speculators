"""Actual loopback safetensor transport, checkpoint identity and backend routing."""

import copy
import json
import pickle
import shutil
import threading
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from hs_connectors.transfer import HttpBackend
from safetensors.torch import save_file

from speculators.remote_http import (
    MANIFEST,
    QwenHiddenStatesServer,
    QwenHttpTrainingTransfer,
    checkpoint_contract,
    save_contract,
)
from speculators.train import data as data_module

TOKEN = "qwen-http-test-only-0123456789abcdef0123456789"


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / "target-model"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "hidden_size": 8,
                "num_hidden_layers": 4,
                "vocab_size": 32,
                "torch_dtype": "bfloat16",
            }
        )
    )
    save_file(
        {"model.embed_tokens.weight": torch.ones(32, 8, dtype=torch.bfloat16)},
        root / "model.safetensors",
    )
    return root


@pytest.fixture
def target(checkpoint, tmp_path):
    directory = tmp_path / "remote-hs"
    contract = checkpoint_contract(checkpoint, [1, 3])
    save_contract(directory, contract)
    server = QwenHiddenStatesServer(("127.0.0.1", 0), directory, TOKEN, training=True)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    yield server, contract, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def client(target, tmp_path, **kwargs):
    _, contract, endpoint = target
    return QwenHttpTrainingTransfer(
        endpoint,
        TOKEN,
        tmp_path / "downloads",
        timeout=0.5,
        expected_contract=contract,
        **kwargs,
    )


def publish(transfer, server, *, dtype=torch.bfloat16, width=8):
    options = transfer.request_kwargs()
    assert options["max_retries"] == 0
    path = server.directory / f"cmpl-{options['request_id']}-0-deadbeef.safetensors"
    sample = {
        "token_ids": torch.arange(5),
        "hidden_states": torch.randn(5, 3, width).to(dtype),
    }
    save_file(sample, path)
    return path, sample


def test_cross_directory_http_download_is_lossless_and_owned(target, tmp_path):
    server, _, _ = target
    transfer = client(target, tmp_path)
    path, sample = publish(transfer, server)
    loaded = transfer.get_generated(str(path))
    for name in sample:
        assert torch.equal(loaded[name], sample[name])
        assert loaded[name].dtype == sample[name].dtype
    assert path.exists()
    transfer.delete(str(path))
    assert not path.exists()
    assert (tmp_path / "downloads" / MANIFEST).is_file()
    assert not list((tmp_path / "downloads").glob("hstrain-*"))


def test_checkpoint_identity_ignores_directory_but_checks_payload(checkpoint, tmp_path):
    other = tmp_path / "trainer-model"
    shutil.copytree(checkpoint, other)
    expected = checkpoint_contract(checkpoint, [1, 3])
    assert checkpoint_contract(other, [1, 3]) == expected
    save_file(
        {"model.embed_tokens.weight": torch.zeros(32, 8, dtype=torch.bfloat16)},
        other / "model.safetensors",
    )
    assert checkpoint_contract(other, [1, 3]) != expected


def test_indexed_checkpoint_hashes_every_shard(checkpoint):
    first = checkpoint / "model-00001-of-00002.safetensors"
    (checkpoint / "model.safetensors").rename(first)
    second = checkpoint / "model-00002-of-00002.safetensors"
    save_file({"lm_head.weight": torch.ones(32, 8)}, second)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": first.name,
                    "lm_head.weight": second.name,
                }
            }
        )
    )
    before = checkpoint_contract(checkpoint, [1, 3])
    save_file({"lm_head.weight": torch.zeros(32, 8)}, second)
    assert checkpoint_contract(checkpoint, [1, 3]) != before


def test_arrow_training_accepts_verified_alias_and_preserves_layer_layout(
    checkpoint, target, tmp_path, monkeypatch
):
    dataset_path = tmp_path / "data"
    encoded = Dataset.from_dict(
        {
            "input_ids": [[0, 1, 2, 3, 4]],
            "loss_mask": [[0, 0, 1, 1, 1]],
            "seq_len": [5],
        }
    )
    encoded.set_format(type="torch")
    encoded.save_to_disk(str(dataset_path))
    transfer = client(target, tmp_path)
    dataset = data_module.ArrowDataset(
        max_len=8, datapath=dataset_path, transfer=transfer, model=str(checkpoint)
    )
    api = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="qwen3-4b-http")])
        )
    )
    monkeypatch.setattr(data_module.openai, "OpenAI", lambda **_: api)
    hidden = torch.randn(5, 3, 8, dtype=torch.bfloat16)
    generated = []

    def generate(api_client, model, item, **options):
        assert api_client is api
        assert model == "qwen3-4b-http"
        assert options["max_retries"] == 0
        path = target[0].directory / f"cmpl-{options['request_id']}-0.safetensors"
        save_file(
            {"hidden_states": hidden, "token_ids": torch.tensor(item["input_ids"])},
            path,
        )
        generated.append(path)
        return str(path)

    monkeypatch.setattr(data_module, "generate_hidden_states", generate)
    row = dataset[0]
    assert torch.equal(row["hidden_states"], hidden[:, :-1].flatten(1))
    assert torch.equal(row["verifier_last_hidden_states"], hidden[:, -1])
    assert row["loss_mask"].tolist() == [0, 0, 1, 1, 1]
    assert not generated[0].exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("auxiliary_hs_ids", [3, 1]),
        ("checkpoint_sha256", "f" * 64),
        ("hidden_size", 16),
    ],
)
def test_target_mismatch_is_rejected_before_generation(target, tmp_path, field, value):
    _, contract, endpoint = target
    expected = copy.deepcopy(contract)
    expected[field] = value
    transfer = QwenHttpTrainingTransfer(
        endpoint, TOKEN, tmp_path / "downloads", expected_contract=expected
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        transfer.request_kwargs()


@pytest.mark.parametrize(("dtype", "width"), [(torch.float32, 8), (torch.bfloat16, 9)])
def test_malformed_payload_remains_remote_and_cannot_be_deleted(
    target, tmp_path, dtype, width
):
    transfer = client(target, tmp_path)
    path, _ = publish(transfer, target[0], dtype=dtype, width=width)
    with pytest.raises(ValueError, match="shape/dtype"):
        transfer.get_generated(str(path))
    with pytest.raises(ValueError, match="successfully"):
        transfer.delete(str(path))
    assert path.exists()


def test_foreign_request_and_changed_contract_are_rejected(target, tmp_path):
    owner = client(target, tmp_path)
    path, _ = publish(owner, target[0])
    other = client(target, tmp_path)
    with pytest.raises(ValueError, match="active training"):
        other.get_generated(str(path))
    changed = copy.deepcopy(target[1])
    changed["checkpoint_sha256"] = "e" * 64
    (target[0].directory / MANIFEST).write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="status 400"):
        owner.get_generated(str(path))
    assert path.exists()


def test_bad_token_rejected_and_worker_pickle_drops_connections(target, tmp_path):
    _, contract, endpoint = target
    wrong = QwenHttpTrainingTransfer(
        endpoint, TOKEN + "x", tmp_path / "wrong", expected_contract=contract
    )
    with pytest.raises(ValueError, match="401"):
        wrong.setup()
    transfer = client(target, tmp_path)
    transfer.request_kwargs()
    child = pickle.loads(pickle.dumps(transfer))  # noqa: S301 -- Own in-process fixture.
    assert not child._pending
    assert not child._clients
    assert child.read_contract() == contract


def test_http_backend_routes_standard_qwen_without_dsv4_flags(
    checkpoint, target, tmp_path, monkeypatch
):
    monkeypatch.setenv("SPECULATORS_HS_HTTP_TOKEN", TOKEN)
    args = SimpleNamespace(
        target_hidden_state_format="standard",
        hs_http_endpoints=[target[2]],
        hidden_states_path=str(tmp_path / "trainer-local"),
        hs_http_timeout=0.5,
        hs_http_max_file_bytes=1024**2,
        on_generate="delete",
        verifier_name_or_path=str(checkpoint),
        target_layer_ids=[1, 3],
    )
    transfer = HttpBackend.from_train_args(args, str(tmp_path))
    assert isinstance(transfer, QwenHttpTrainingTransfer)
    assert transfer.allow_model_name_relocation
    assert transfer.read_contract() == target[1]
    args.on_generate = "cache"
    with pytest.raises(ValueError, match="on-generate delete"):
        HttpBackend.from_train_args(args, str(tmp_path))


def test_existing_directory_cannot_be_relabelled(target):
    changed = copy.deepcopy(target[1])
    changed["auxiliary_hs_ids"] = [1, 2]
    with pytest.raises(ValueError, match="another model/layer"):
        save_contract(target[0].directory, changed)


@pytest.mark.parametrize("matching", [True, False])
def test_two_target_hosts_must_match_before_local_download(target, tmp_path, matching):
    peer_path = tmp_path / "peer-hs"
    peer_contract = copy.deepcopy(target[1])
    if not matching:
        peer_contract["checkpoint_sha256"] = "a" * 64
    save_contract(peer_path, peer_contract)
    with QwenHiddenStatesServer(
        ("127.0.0.1", 0), peer_path, TOKEN, training=True
    ) as peer:
        thread = threading.Thread(
            target=peer.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            transfer = QwenHttpTrainingTransfer(
                [target[2], f"http://127.0.0.1:{peer.server_port}"],
                TOKEN,
                tmp_path / "downloads",
                expected_contract=target[1],
            )
            if matching:
                path, sample = publish(transfer, target[0])
                loaded = transfer.get_generated(str(path))
                assert torch.equal(loaded["hidden_states"], sample["hidden_states"])
                transfer.delete(str(path))
                assert not path.exists()
                assert not list(peer_path.glob("*.safetensors"))
            else:
                with pytest.raises(ValueError, match="contract mismatch"):
                    transfer.request_kwargs()
        finally:
            peer.shutdown()
            thread.join(timeout=2)
