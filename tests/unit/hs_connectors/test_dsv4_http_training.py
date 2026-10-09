"""Real BF16 safetensors transfer over loopback HTTP; no target/NPU needed."""

import json
import threading

import pytest
import torch
from safetensors.torch import save_file

from speculators_dsv4 import hs_profile
from speculators_dsv4.contract import DEFAULT_LAYERS, MANIFEST, make_manifest
from speculators_dsv4.hs_http_server import HiddenStatesServer
from speculators_dsv4.hs_training_http import HttpTrainingTransfer


@pytest.mark.parametrize("profile_every", [0, 1])
def test_http_training_preserves_bf16_payload_and_token_ids(
    tmp_path, monkeypatch, capsys, profile_every
):
    monkeypatch.setenv("SPECULATORS_HS_PROFILE_EVERY", str(profile_every))
    monkeypatch.setattr(hs_profile, "_STATE", None)
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
            file_bytes = path.stat().st_size

            @hs_profile.profile_sample
            def download():
                with hs_profile.profile_stage("get_generated_ms"):
                    return transfer.get_generated(str(path))

            loaded = download()
            for name, value in original.items():
                assert loaded[name].dtype == value.dtype
                assert torch.equal(loaded[name], value)
            transfer.delete(str(path))
            assert not path.exists()
            assert list((tmp_path / "trainer-local").iterdir()) == []
            records = [
                json.loads(line)
                for line in capsys.readouterr().err.splitlines()
                if line.startswith('{"event":"hs_profile"')
            ]
            assert len(records) == profile_every
            if profile_every:
                stats = records[0]["stats"]
                assert {
                    "http_route_ms",
                    "http_download_ms",
                    "http_load_ms",
                } <= stats.keys()
                assert stats["http_file_bytes"]["latest"] == file_bytes
                assert token not in json.dumps(records)
                assert str(path) not in json.dumps(records)
        finally:
            server.shutdown()
            thread.join(timeout=2)
