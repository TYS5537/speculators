"""Shared-contract publication tests; these do not certify real NFS locking."""

# ruff: noqa: PT009, PT027 -- Dependency-light unittest suite.

import json
import os
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from speculators_dsv4.contract import MANIFEST, ensure_manifest, wait_for_manifest


class SharedManifestTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "shared-hs"
        self.expected = {"format": "fixture", "checkpoint_signature": "checkpoint-a"}
        self.runtime = {"method": None}

    def publish(self):
        ensure_manifest(
            self.root, self.expected, create=True, runtime_quantization=self.runtime
        )

    def wait(self, **kwargs):
        wait_for_manifest(
            self.root, self.expected, runtime_quantization=self.runtime, **kwargs
        )

    def test_json_is_complete_before_manifest_becomes_visible(self):
        link = os.link

        def publish_complete(source, destination):
            self.assertFalse(destination.exists())
            self.assertEqual(
                json.loads(source.read_text(encoding="utf-8")),
                {**self.expected, "runtime_quantization": self.runtime},
            )
            link(source, destination)

        with patch("speculators_dsv4.contract.os.link", side_effect=publish_complete):
            self.publish()
        self.assertEqual([path.name for path in self.root.iterdir()], [MANIFEST])
        self.wait()

    @unittest.skipIf(os.name == "nt", "POSIX umask permissions need a POSIX filesystem")
    def test_publication_preserves_normal_shared_file_permissions(self):
        reference = self.root.parent / "normal-file"
        with reference.open("x", encoding="utf-8") as stream:
            stream.write("fixture")
        self.publish()
        self.assertEqual(
            stat.S_IMODE((self.root / MANIFEST).stat().st_mode),
            stat.S_IMODE(reference.stat().st_mode),
        )

    def test_worker_started_first_waits_for_head_without_writing(self):
        waiting = threading.Event()
        published = threading.Event()

        def wait_for_head(_seconds):
            waiting.set()
            if not published.wait(timeout=5):
                raise TimeoutError("Test head did not publish")

        with (
            patch("speculators_dsv4.contract.time.sleep", side_effect=wait_for_head),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            worker = executor.submit(self.wait, timeout=10)
            try:
                self.assertTrue(waiting.wait(timeout=5))
                self.assertFalse(self.root.exists())
                self.publish()
            finally:
                published.set()
            worker.result(timeout=5)

    def test_worker_timeout_never_creates_local_directory(self):
        with (
            patch("speculators_dsv4.contract.time.monotonic", side_effect=[0, 2]),
            self.assertRaisesRegex(TimeoutError, "same shared HS directory"),
        ):
            self.wait(timeout=1)
        self.assertFalse(self.root.exists())

    def test_timeout_must_be_finite_and_positive(self):
        for timeout in (0, -1, float("nan"), float("inf")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                self.wait(timeout=timeout)
        self.assertFalse(self.root.exists())

    def test_worker_rejects_mismatched_contract_without_rewriting(self):
        self.publish()
        path = self.root / MANIFEST
        original = path.read_bytes()
        for expected, runtime in (
            ({**self.expected, "checkpoint_signature": "checkpoint-b"}, self.runtime),
            (self.expected, {"method": "ascend"}),
        ):
            with self.subTest(expected=expected, runtime=runtime):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    wait_for_manifest(self.root, expected, runtime_quantization=runtime)
                self.assertEqual(path.read_bytes(), original)

    def test_worker_rejects_corrupt_contract_immediately(self):
        self.root.mkdir()
        (self.root / MANIFEST).write_text("{", encoding="utf-8")
        with (
            patch("speculators_dsv4.contract.time.sleep") as sleep,
            self.assertRaises(json.JSONDecodeError),
        ):
            self.wait()
        sleep.assert_not_called()

    def test_racing_publisher_cannot_overwrite_different_contract(self):
        link = os.link
        other = {**self.expected, "checkpoint_signature": "other"}

        def racing_link(source, destination):
            destination.write_text(json.dumps(other), encoding="utf-8")
            link(source, destination)

        with (
            patch("speculators_dsv4.contract.os.link", side_effect=racing_link),
            self.assertRaisesRegex(ValueError, "mismatch"),
        ):
            self.publish()
        self.assertEqual(json.loads((self.root / MANIFEST).read_text()), other)
        self.assertEqual([path.name for path in self.root.iterdir()], [MANIFEST])

    def test_racing_identical_publisher_can_reuse_complete_contract(self):
        link = os.link

        def racing_link(source, destination):
            destination.write_bytes(source.read_bytes())
            link(source, destination)

        with patch("speculators_dsv4.contract.os.link", side_effect=racing_link):
            self.publish()
        self.wait()
        self.assertEqual([path.name for path in self.root.iterdir()], [MANIFEST])

    def test_failed_publication_cleans_only_its_temporary_file(self):
        with (
            patch(
                "speculators_dsv4.contract.os.link", side_effect=OSError("unsupported")
            ),
            self.assertRaisesRegex(OSError, "unsupported"),
        ):
            self.publish()
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
