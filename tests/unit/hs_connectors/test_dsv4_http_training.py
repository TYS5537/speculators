"""Real BF16 safetensors transfer over loopback HTTP; no target/NPU needed."""

import json
import threading

import torch
from safetensors.torch import save_file

from speculators_dsv4.contract import DEFAULT_LAYERS, MANIFEST, make_manifest
from speculators_dsv4.hs_http_server import HiddenStatesServer
from speculators_dsv4.hs_training_http import HttpTrainingTransfer


def test_http_training_preserves_bf16_payload_and_token_ids(tmp_path):
    remote = tmp_path / "target-local"
    remote.mkdir()
    contract = {
        **make_manifest(
            {"model_path": "/target/model", "checkpoint_signature": "a" * 64},
            DEFAULT_LAYERS,
        ),
        "runtime_quantization": {"method": None},
    }
    (remote / MANIFEST).write_text(json.dumps(contract), encoding="utf-8")
    token = "fixture-only-0123456789abcdef0123456789abcdef"
    with HiddenStatesServer(("127.0.0.1", 0), remote, token, training=True) as server:
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            transfer = HttpTrainingTransfer(
                [f"http://127.0.0.1:{server.server_port}"],
                token,
                tmp_path / "trainer-local",
                expected_contract={**contract, "model_path": "/trainer/model"},
            )
            nonce = transfer.request_kwargs()["request_id"]
            path = remote / f"cmpl-{nonce}-0-deadbeef.safetensors"
            original = {
                "hidden_states": torch.randn(7, 6, 4096, dtype=torch.bfloat16),
                "token_ids": torch.arange(7, dtype=torch.int64),
            }
            save_file(original, path)
            loaded = transfer.get_generated(str(path))
            for name, value in original.items():
                assert loaded[name].dtype == value.dtype
                assert torch.equal(loaded[name], value)
            transfer.delete(str(path))
            assert not path.exists()
            assert list((tmp_path / "trainer-local").iterdir()) == []
        finally:
            server.shutdown()
            thread.join(timeout=2)
