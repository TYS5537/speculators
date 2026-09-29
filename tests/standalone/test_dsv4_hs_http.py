"""Real loopback HTTP tests; no vLLM, PyTorch, NPU or external service required."""

# ruff: noqa: PT009, PT027 -- Also runnable with unittest alone.
# ruff: noqa: SIM117 -- Keep expected exceptions separate from the tested context.

import hashlib
import io
import json
import os
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from speculators_dsv4 import HS_FORMAT, hs_http_server
from speculators_dsv4.contract import MANIFEST
from speculators_dsv4.hs_http import HttpHiddenStates, validate_endpoint, validate_token
from speculators_dsv4.hs_http_server import HiddenStatesServer

TOKEN = "test-only-token-0123456789abcdef0123456789"


class HttpTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.local = self.root / "local"
        self.manifest = {
            "format": HS_FORMAT,
            "checkpoint_signature": "fixture",
            "model_path": "/server/models/dsv4",
        }
        (self.remote / MANIFEST).write_text(json.dumps(self.manifest))
        self.server = HiddenStatesServer(("127.0.0.1", 0), self.remote, TOKEN)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.client = HttpHiddenStates(self.endpoint, TOKEN, self.local, timeout=0.3)
        self.client.validate_manifest(self.manifest)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def artifact(self, content=b"binary-hs\x00\x01"):
        request_id = self.client.new_request_id()
        path = self.remote / f"cmpl-{request_id}-0-deadbeef.safetensors"
        path.write_bytes(content)
        return request_id, path

    def request(self, route, *, method="GET", token=TOKEN):
        opener = build_opener(ProxyHandler({}))
        return opener.open(
            Request(  # noqa: S310 -- Local fixture HTTP only.
                self.endpoint + route,
                method=method,
                headers={"Authorization": "Bearer " + token},
            ),
            timeout=1,
        )

    def test_stream_bytes_and_delete_only_owned_remote_artifact(self):
        request_id, path = self.artifact()
        training = self.remote / "hs_0.safetensors"
        training.write_bytes(b"training-cache")
        with self.client.artifact(str(path), request_id) as downloaded:
            self.assertEqual(downloaded.read_bytes(), path.read_bytes())
            self.assertNotEqual(downloaded.parent, path.parent)
        self.assertFalse(path.exists())
        self.assertEqual(training.read_bytes(), b"training-cache")
        self.assertEqual(list(self.local.iterdir()), [])

    def test_keep_preserves_remote_file_but_not_local_temporary_copy(self):
        request_id, path = self.artifact()
        with self.client.artifact(str(path), request_id, keep=True) as downloaded:
            self.assertTrue(downloaded.exists())
        self.assertTrue(path.exists())
        self.assertEqual(list(self.local.iterdir()), [])

    def test_checksum_and_body_use_one_snapshot_even_if_source_changes(self):
        original = b"first-version-00"
        changed = b"other-version-00"
        request_id, path = self.artifact(original)
        end_headers = hs_http_server._Handler.end_headers

        def change_after_headers(handler):
            if handler.path == "/v1/files/" + path.name:
                path.write_bytes(changed)
            end_headers(handler)

        with patch.object(hs_http_server._Handler, "end_headers", change_after_headers):
            with self.client.artifact(str(path), request_id, keep=True) as downloaded:
                self.assertEqual(downloaded.read_bytes(), original)
        self.assertEqual(path.read_bytes(), changed)

    def changing_source(self, path, content):
        regular_file = hs_http_server._regular_file
        changed = False

        @contextmanager
        def open_file(candidate, limit):
            nonlocal changed
            with regular_file(candidate, limit) as stream:
                read = stream.read

                def read_and_change(size):
                    nonlocal changed
                    chunk = read(size)
                    if not changed:
                        changed = True
                        path.write_bytes(content)
                    return chunk

                if candidate == path:
                    with patch.object(stream, "read", side_effect=read_and_change):
                        yield stream
                else:
                    yield stream

        return patch.object(hs_http_server, "_regular_file", open_file)

    def test_source_change_during_snapshot_returns_pending_before_headers(self):
        _, path = self.artifact(b"original")
        with self.changing_source(path, b"changed-longer"):
            with self.assertRaises(HTTPError) as caught:
                self.request("/v1/files/" + path.name)
            self.assertEqual(caught.exception.code, 409)
            caught.exception.close()
        self.assertTrue(path.exists())

    def test_client_retries_same_artifact_after_source_stabilizes(self):
        request_id, path = self.artifact(b"original")
        changed = b"changed-longer"
        with self.changing_source(path, changed):
            with self.client.artifact(str(path), request_id, keep=True) as downloaded:
                self.assertEqual(downloaded.read_bytes(), changed)
        self.assertTrue(path.exists())

    def test_snapshot_growth_is_bounded_by_file_limit(self):
        _, path = self.artifact(b"small")
        self.server.max_file_bytes = 8
        with self.changing_source(path, b"x" * 100):
            with self.assertRaises(HTTPError) as caught:
                self.request("/v1/files/" + path.name)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        self.assertTrue(path.exists())

    def test_snapshot_spills_to_disk_and_closes_after_download(self):
        request_id, path = self.artifact(b"snapshot-data" * 100)
        snapshots = []
        closed = threading.Event()
        spool = tempfile.SpooledTemporaryFile

        @contextmanager
        def create_snapshot(*args, **kwargs):
            with spool(*args, **kwargs) as snapshot:
                snapshots.append(snapshot)
                yield snapshot
            closed.set()

        with (
            patch.object(hs_http_server, "SNAPSHOT_MEMORY_BYTES", 8),
            patch.object(
                hs_http_server.tempfile, "SpooledTemporaryFile", create_snapshot
            ),
        ):
            with self.client.artifact(str(path), request_id, keep=True) as downloaded:
                self.assertEqual(downloaded.read_bytes(), path.read_bytes())
        self.assertEqual(len(snapshots), 1)
        self.assertTrue(snapshots[0]._rolled)
        # The client can finish reading just before the server closes its snapshot.
        self.assertTrue(closed.wait(timeout=2))
        self.assertTrue(snapshots[0].closed)

    def test_send_failure_does_not_append_an_http_error_to_binary_body(self):
        _, path = self.artifact()

        def interrupted(handler, path, *, delete):
            handler.send_response(200)
            handler.send_header("Content-Length", "1024")
            handler.end_headers()
            handler.wfile.write(b"partial")
            raise OSError("interrupted send")

        with patch.object(hs_http_server._Handler, "_artifact", interrupted):
            with self.request("/v1/files/" + path.name) as response:
                self.assertEqual(response.read(1024), b"partial")
        self.assertTrue(path.exists())

    def test_intermediate_reference_request_does_not_download(self):
        request_id, path = self.artifact()
        with patch.object(self.client, "_download", side_effect=AssertionError):
            with self.client.artifact(
                str(path), request_id, download=False
            ) as downloaded:
                self.assertIsNone(downloaded)
        self.assertFalse(path.exists())

    def test_authentication_required_for_manifest_download_and_delete(self):
        _, path = self.artifact()
        for method, route in (
            ("GET", "/v1/manifest"),
            ("GET", "/v1/files/" + path.name),
            ("DELETE", "/v1/files/" + path.name),
        ):
            with (
                self.subTest(method=method, route=route),
                self.assertRaises(HTTPError) as caught,
            ):
                self.request(route, method=method, token="wrong")
            self.assertEqual(caught.exception.code, 401)
            caught.exception.close()
        self.assertTrue(path.exists())

    def test_training_files_traversal_queries_and_directory_listing_are_rejected(self):
        training = self.remote / "hs_0.safetensors"
        training.write_bytes(b"untouched")
        for name in (
            "hs_0.safetensors",
            "cmpl-ordinary-0.safetensors",
            "../" + MANIFEST,
            "%2e%2e%2f" + MANIFEST,
            "",
            "x?name=hs_0.safetensors",
        ):
            for method in ("GET", "DELETE"):
                with (
                    self.subTest(name=name, method=method),
                    self.assertRaises(HTTPError) as caught,
                ):
                    self.request("/v1/files/" + name, method=method)
                self.assertEqual(caught.exception.code, 400)
                caught.exception.close()
        self.assertEqual(training.read_bytes(), b"untouched")

    def test_wrong_request_or_remote_directory_is_rejected_before_network_io(self):
        request_id, path = self.artifact()
        for handle, identity in (
            (str(path), self.client.new_request_id()),
            (str(self.root / path.name), request_id),
            (str(self.remote / ".." / path.name), request_id),
        ):
            with self.subTest(handle=handle), self.assertRaises(ValueError):
                with self.client.artifact(handle, identity):
                    self.fail("Unexpected download")
        self.assertTrue(path.exists())

    def test_oversized_and_hardlinked_files_are_rejected_without_deletion(self):
        request_id, path = self.artifact()
        self.server.max_file_bytes = 1
        with self.assertRaisesRegex(ValueError, "status 400"):
            with self.client.artifact(str(path), request_id):
                self.fail("Unexpected download")
        self.server.max_file_bytes = 1024
        alias = self.root / "hardlink"
        os.link(path, alias)
        with self.assertRaisesRegex(ValueError, "status 400"):
            with self.client.artifact(str(path), request_id):
                self.fail("Unexpected download")
        self.assertTrue(path.exists())
        self.assertTrue(alias.exists())

    def test_symlink_file_and_symlink_lock_are_rejected(self):
        request_id, path = self.artifact()
        unrelated = self.root / "unrelated"
        unrelated.write_bytes(b"untouched")
        path.unlink()
        try:
            path.symlink_to(unrelated)
        except OSError:
            self.skipTest("Host cannot create symlinks")
        for method in ("GET", "DELETE"):
            with self.assertRaises(HTTPError) as caught:
                self.request("/v1/files/" + path.name, method=method)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        path.unlink()
        path.write_bytes(b"hs")
        Path(str(path) + ".lock").symlink_to(unrelated)
        with self.assertRaisesRegex(ValueError, "status 400"):
            with self.client.artifact(str(path), request_id):
                self.fail("Unexpected download")
        self.assertEqual(unrelated.read_bytes(), b"untouched")

    def test_busy_writer_times_out_without_reading_or_deleting(self):
        request_id, path = self.artifact()
        with patch(
            "speculators_dsv4.hs_http_server._writer_lock", side_effect=BlockingIOError
        ):
            with self.assertRaises(TimeoutError):
                with self.client.artifact(str(path), request_id):
                    self.fail("Unexpected download")
        self.assertTrue(path.exists())
        self.assertEqual(list(self.local.iterdir()), [])

    @unittest.skipUnless(os.name == "posix", "Real producer locks require POSIX")
    def test_real_writer_lock_waits_for_completion(self):
        import fcntl  # noqa: PLC0415 -- POSIX-only test; Windows collection must work.

        request_id, path = self.artifact()
        lock = Path(str(path) + ".lock")
        with lock.open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            with self.assertRaises(HTTPError) as caught:
                self.request("/v1/files/" + path.name)
            self.assertEqual(caught.exception.code, 409)
            caught.exception.close()
        with self.client.artifact(str(path), request_id):
            pass
        self.assertFalse(lock.exists())

    def test_checksum_truncation_and_client_size_limit_preserve_remote_file(self):
        request_id, path = self.artifact()
        for length, digest in (
            (4, "0" * 64),
            (100, hashlib.sha256(b"data").hexdigest()),
            (self.client.max_file_bytes + 1, "0" * 64),
        ):
            response = io.BytesIO(b"data")
            response.headers = {"Content-Length": str(length), "X-HS-SHA256": digest}
            with (
                patch.object(self.client, "_open", return_value=response) as opened,
                self.assertRaises(ValueError) as caught,
            ):
                with self.client.artifact(str(path), request_id):
                    self.fail("Unexpected download")
            opened.assert_called_once_with("/v1/files/" + path.name)
            self.assertTrue(path.exists())
            if length <= self.client.max_file_bytes:
                self.assertIn(path.name, str(caught.exception))
                self.assertIn(digest, str(caught.exception))
                self.assertIn(
                    hashlib.sha256(b"data").hexdigest(), str(caught.exception)
                )
                metadata = list(self.local.glob("failed-downloads/*/error.json"))
                self.assertEqual(len(metadata), 1 if length == 4 else 2)
                for item in metadata:
                    saved = json.loads(item.read_text())
                    self.assertEqual(saved["artifact"], path.name)
                    self.assertEqual(saved["received_bytes"], 4)
                    self.assertEqual((item.parent / path.name).read_bytes(), b"data")
                    self.assertNotIn(TOKEN, item.read_text())
            self.assertFalse(list(self.local.glob("hshttp-*")))

    def test_failure_evidence_error_does_not_mask_checksum_error(self):
        request_id, path = self.artifact()
        response = io.BytesIO(b"data")
        response.headers = {"Content-Length": "4", "X-HS-SHA256": "0" * 64}
        with (
            patch.object(self.client, "_open", return_value=response),
            patch.object(Path, "rename", side_effect=OSError("cannot save evidence")),
            self.assertLogs("speculators_dsv4.hs_http", level="WARNING"),
            self.assertRaisesRegex(ValueError, "HS HTTP checksum mismatch"),
        ):
            with self.client.artifact(str(path), request_id):
                self.fail("Unexpected download")
        self.assertTrue(path.exists())

    def test_changed_manifest_or_mismatched_checkpoint_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.client.validate_manifest(
                {**self.manifest, "checkpoint_signature": "other"}
            )
        request_id, path = self.artifact()
        (self.remote / MANIFEST).write_text(
            json.dumps({**self.manifest, "changed": True})
        )
        with self.assertRaisesRegex(ValueError, "status 400"):
            with self.client.artifact(str(path), request_id):
                self.fail("Unexpected download")
        self.assertTrue(path.exists())

    def test_model_path_can_differ_but_remaining_identity_is_strict(self):
        expected = {**self.manifest, "model_path": str(self.root / "local-model")}
        original = (self.remote / MANIFEST).read_bytes()
        actual = self.client.validate_manifest(expected)
        self.assertEqual(actual["model_path"], "/server/models/dsv4")
        for changed in (
            {"checkpoint_signature": "wrong"},
            {"format": "wrong"},
            {"auxiliary_hs_ids": [1, 2]},
            {"unexpected": True},
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "mismatch"),
            ):
                self.client.validate_manifest({**expected, **changed})
        self.assertEqual((self.remote / MANIFEST).read_bytes(), original)

    def test_invalid_endpoints_and_tokens(self):
        for endpoint in (
            "file:///tmp/x",
            "http://user:secret@host",
            "http://host?key=x",
            "http://host/#fragment",
            "http://host\n",
            "host:123",
            None,
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                validate_endpoint(endpoint)
        for token in (None, "", "short", "x" * 32 + "\n", "x" * 32 + "汉"):
            with self.assertRaises(ValueError):
                validate_token(token)

    def test_cleanup_failure_does_not_mask_consumer_error(self):
        request_id, path = self.artifact()
        with (
            patch.object(self.client, "_delete", side_effect=OSError("offline")),
            self.assertLogs("speculators_dsv4.hs_http", level="WARNING"),
            self.assertRaisesRegex(ValueError, "consumer failed"),
        ):
            with self.client.artifact(str(path), request_id):
                raise ValueError("consumer failed")
        self.assertTrue(path.exists())
        self.assertEqual(list(self.local.iterdir()), [])

    def test_missing_file_is_pending_even_for_cleanup(self):
        request_id, path = self.artifact()
        path.unlink()
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/files/" + path.name, method="DELETE")
        self.assertEqual(caught.exception.code, 409)
        caught.exception.close()

    def test_redirects_do_not_forward_authorization(self):
        def redirect(handler):
            handler.send_response(302)
            handler.send_header("Location", self.endpoint + "/v1/files/forbidden")
            handler.send_header("Content-Length", "0")
            handler.end_headers()

        with patch("speculators_dsv4.hs_http_server._Handler._manifest", redirect):
            with self.assertRaisesRegex(ValueError, "redirects are disabled"):
                self.client.validate_manifest(self.manifest)

    def test_concurrent_downloads_use_distinct_local_files(self):
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        pairs = [self.artifact(bytes([index]) * 100) for index in range(8)]

        def download(pair):
            request_id, path = pair
            expected = path.read_bytes()
            with self.client.artifact(str(path), request_id) as local:
                self.assertEqual(local.read_bytes(), expected)
                return local.parent

        with ThreadPoolExecutor(max_workers=4) as pool:
            directories = list(pool.map(download, pairs))
        self.assertEqual(len(set(directories)), 8)
        self.assertTrue(all(not path.exists() for _, path in pairs))
        self.assertEqual(list(self.local.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
