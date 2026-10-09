"""Loopback HTTP and fake-store tests; not NPU/RDMA certification."""

# ruff: noqa: PT009, PT027 -- Also runs without pytest or torch.

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_dsv4.contract import DEFAULT_LAYERS, MANIFEST, make_manifest
from speculators_dsv4.hs_http_server import HiddenStatesServer
from speculators_dsv4.hs_training_http import HttpTrainingTransfer
from speculators_dsv4.mooncake_training import (
    Dsv4MooncakeTransfer,
    contract_key,
    sample_metadata,
    validate_namespace,
)
from speculators_dsv4.training import _read_remote_contract
from speculators_dsv4.training_contract import bind_remote_training_contract

TOKEN = "fixture-only-0123456789abcdef0123456789abcdef"


def contract():
    return {
        **make_manifest(
            {"model_path": "/target/model", "checkpoint_signature": "a" * 64},
            DEFAULT_LAYERS,
        ),
        "runtime_quantization": {"method": "ascend"},
    }


class HttpTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = contract()
        self.servers = []
        self.endpoints = []
        for name in ("first", "second"):
            folder = self.root / name
            folder.mkdir()
            (folder / MANIFEST).write_text(json.dumps(self.manifest), encoding="utf-8")
            server = HiddenStatesServer(("127.0.0.1", 0), folder, TOKEN, training=True)
            thread = threading.Thread(
                target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
            )
            thread.start()
            self.addCleanup(self.stop, server, thread)
            self.servers.append(server)
            self.endpoints.append(f"http://127.0.0.1:{server.server_port}")
        self.client = self.transfer()

    @staticmethod
    def stop(server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def transfer(self, **kwargs):
        return HttpTrainingTransfer(
            self.endpoints, TOKEN, self.root / "downloads", timeout=0.2, **kwargs
        )

    def artifact(self, host=1):
        request = self.client.request_kwargs()
        self.assertEqual(request["max_retries"], 0)
        path = (
            self.servers[host].directory
            / f"cmpl-{request['request_id']}-0-deadbeef.safetensors"
        )
        path.write_bytes(b"verified-fixture")
        return path

    def test_two_hosts_contract_and_trainer_path_relocation(self):
        expected = {**self.manifest, "model_path": "/trainer/model"}
        self.assertEqual(
            self.transfer(expected_contract=expected).read_contract(), self.manifest
        )

    def test_routes_same_directory_name_to_only_owner_and_deletes_after_success(self):
        path = self.artifact()
        # Different loopback fixtures emulate the same absolute directory on two hosts.
        self.client._clients[0].remote_directory = self.client._clients[
            1
        ].remote_directory
        with patch(
            "speculators_dsv4.hs_training_http._load_tensors",
            side_effect=lambda value: value.read_bytes(),
        ):
            self.assertEqual(self.client.get_generated(str(path)), b"verified-fixture")
        self.assertTrue(path.exists())
        self.client.delete(str(path))
        self.assertFalse(path.exists())
        self.assertEqual(list((self.root / "downloads").iterdir()), [])
        with self.assertRaisesRegex(ValueError, "successfully"):
            self.client.delete(str(path))

    def test_other_nonce_cannot_be_read_or_deleted(self):
        path = self.artifact()
        other = self.transfer()
        other.setup()
        with self.assertRaisesRegex(ValueError, "active training"):
            other.get_generated(str(path))
        with self.assertRaisesRegex(ValueError, "successfully"):
            other.delete(str(path))
        self.assertTrue(path.exists())

    def test_safetensor_decode_failure_retains_remote_artifact(self):
        path = self.artifact()
        with (
            patch(
                "speculators_dsv4.hs_training_http._load_tensors",
                side_effect=ValueError("corrupt tensor"),
            ),
            self.assertRaisesRegex(ValueError, "corrupt tensor"),
        ):
            self.client.get_generated(str(path))
        self.assertTrue(path.exists())
        with self.assertRaisesRegex(ValueError, "successfully"):
            self.client.delete(str(path))

    def test_eval_sidecar_is_rejected(self):
        self.servers[0].training = False
        with self.assertRaisesRegex(ValueError, "--training"):
            self.client.setup()

    def test_target_quantization_disagreement_is_rejected(self):
        altered = {**self.manifest, "runtime_quantization": {"method": None}}
        server = self.servers[1]
        server.manifest = altered
        (server.directory / MANIFEST).write_text(json.dumps(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.client.setup()

    def test_server_restart_with_other_identity_cannot_deliver_sample(self):
        path = self.artifact()
        server = self.servers[1]
        server.manifest = {**self.manifest, "checkpoint_signature": "b" * 64}
        (server.directory / MANIFEST).write_text(
            json.dumps(server.manifest), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "status 400|contract changed"):
            self.client.get_generated(str(path))
        self.assertTrue(path.exists())

    def test_missing_artifact_timeout_never_changes_other_files(self):
        path = self.artifact()
        path.unlink()
        sentinel = path.parent / "hs_100.safetensors"
        sentinel.write_bytes(b"untouched")
        with self.assertRaises(TimeoutError):
            self.client.get_generated(str(path))
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_changed_contract_cannot_delete_previously_downloaded_artifact(self):
        path = self.artifact()
        with patch("speculators_dsv4.hs_training_http._load_tensors", return_value={}):
            self.client.get_generated(str(path))
        server = self.servers[1]
        server.manifest = {**self.manifest, "checkpoint_signature": "b" * 64}
        (server.directory / MANIFEST).write_text(
            json.dumps(server.manifest), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "status 400"):
            self.client.delete(str(path))
        self.assertTrue(path.exists())

    def test_ambiguous_owner_is_rejected(self):
        path = self.artifact()
        (self.servers[0].directory / path.name).write_bytes(path.read_bytes())
        self.client._clients[0].remote_directory = self.client._clients[
            1
        ].remote_directory
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            self.client.get_generated(str(path))

    def test_serializing_client_drops_live_request_ownership(self):
        self.artifact()
        state = self.client.__getstate__()
        self.assertFalse(state["_pending"])
        self.assertFalse(state["_clients"])


class FakeStore:
    def __init__(self, namespace, dp_size):
        self.is_setup = False
        self.contracts = {
            contract_key(namespace, rank): {
                "namespace": namespace,
                "dp_rank": rank,
                "dp_size": dp_size,
                "contract": contract(),
            }
            for rank in range(dp_size)
        }
        self.samples = {}
        self.close = Mock(side_effect=self._close)
        self.get_sample = Mock(side_effect=self._get_sample)
        self.delete_sample = Mock(side_effect=self.samples.pop)

    def setup(self):
        self.is_setup = True

    def _close(self):
        self.is_setup = False

    def get_json(self, key, **kwargs):
        if key not in self.contracts:
            raise TimeoutError("producer missing")
        return copy.deepcopy(self.contracts[key])

    def _get_sample(self, key, *, expected_metadata, timeout):
        sample, metadata = self.samples[key]
        if metadata != expected_metadata:
            raise ValueError("sample contract mismatch")
        return sample


class MooncakeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.namespace = "fixture-run-001"
        self.store = FakeStore(self.namespace, 4)
        self.client = Dsv4MooncakeTransfer(
            self.store, namespace=self.namespace, target_dp_size=4
        )

    def artifact(self):
        request = self.client.request_kwargs()
        key = f"dsv4hs:{self.namespace}:3:cmpl-{request['request_id']}-0-deadbeef"
        self.store.samples[key] = (
            {"fixture": 1},
            sample_metadata(self.namespace, contract()),
        )
        return key

    def test_all_four_producers_and_no_files(self):
        self.assertEqual(self.client.read_contract(), contract())
        key = self.artifact()
        self.assertEqual(self.client.get_generated(key), {"fixture": 1})
        self.client.delete(key)
        self.assertFalse(self.store.samples)
        self.client.close()
        self.assertFalse(self.store.is_setup)

    def test_wrong_dp_size_fails_preflight(self):
        self.client.target_dp_size = 2
        with self.assertRaisesRegex(ValueError, "topology"):
            self.client.read_contract()
        self.store.close.assert_called_once()

    def test_missing_producer_stops_training(self):
        self.store.contracts.pop(contract_key(self.namespace, 3))
        with self.assertRaises(TimeoutError):
            self.client.read_contract()

    def test_one_producer_wrong_quantization_is_rejected(self):
        self.store.contracts[contract_key(self.namespace, 3)]["contract"][
            "runtime_quantization"
        ] = {"method": None}
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.client.read_contract()

    def test_per_sample_identity_mismatch_retains_source(self):
        key = self.artifact()
        self.store.samples[key] = ({"fixture": 1}, {"wrong": "identity"})
        with self.assertRaisesRegex(ValueError, "contract mismatch"):
            self.client.get_generated(key)
        with self.assertRaisesRegex(ValueError, "successfully"):
            self.client.delete(key)
        self.store.delete_sample.assert_not_called()

    def test_other_run_and_unknown_request_rejected_before_read(self):
        key = self.artifact()
        for other in (
            key.replace(self.namespace, "other-run-001"),
            key.replace("hsmc-", "foreign-"),
        ):
            with self.assertRaisesRegex(ValueError, "does not belong"):
                self.client.get_generated(other)
        self.store.get_sample.assert_not_called()

    def test_forked_live_client_is_rejected(self):
        self.client.setup()
        self.client._pid = -1
        with self.assertRaisesRegex(RuntimeError, "live Mooncake"):
            self.client.setup()

    def test_invalid_namespaces(self):
        for value in (None, "short", "../other-run", "run:foreign", "a" * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_namespace(value)


class RemotePreflightTests(unittest.TestCase):
    def test_only_remote_model_path_can_be_relocated(self):
        report = {"model_path": "/trainer/model", "checkpoint_signature": "a" * 64}
        self.assertEqual(
            bind_remote_training_contract(contract(), report, DEFAULT_LAYERS)[
                "model_path"
            ],
            "/trainer/model",
        )
        with self.assertRaisesRegex(ValueError, "mismatch"):
            bind_remote_training_contract(
                contract(), {**report, "checkpoint_signature": "b" * 64}, DEFAULT_LAYERS
            )

    def test_preflight_always_closes_before_workers(self):
        for failure in (None, ValueError("contract mismatch")):
            transfer = Mock(
                read_contract=Mock(return_value=contract(), side_effect=failure)
            )
            backend = SimpleNamespace(from_train_args=Mock(return_value=transfer))
            fake = SimpleNamespace(
                HiddenStatesBackend=SimpleNamespace(registry={"mooncake": backend})
            )
            args = SimpleNamespace(
                hidden_states_backend="mooncake",
                data_path="data",
                on_missing="generate",
                on_generate="delete",
                target_layer_ids=list(DEFAULT_LAYERS),
            )
            with patch.dict("sys.modules", {"hs_connectors": fake}):
                if failure:
                    with self.assertRaises(ValueError):
                        _read_remote_contract(
                            args,
                            {
                                "model_path": "/trainer/model",
                                "checkpoint_signature": "a" * 64,
                            },
                        )
                else:
                    _read_remote_contract(
                        args,
                        {
                            "model_path": "/trainer/model",
                            "checkpoint_signature": "a" * 64,
                        },
                    )
            transfer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
