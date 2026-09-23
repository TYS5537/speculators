"""Relocating evaluation checkpoints must not relax training or weight identity."""

# ruff: noqa: PT009, PT027 -- Also runnable without pytest/torch.

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import (
    DEFAULT_LAYERS,
    MANIFEST,
    ensure_manifest,
    make_manifest,
)
from speculators_dsv4.eval_contract import (
    bind_draft_verifier,
    read_eval_manifest,
    validate_draft_target,
    validate_eval_manifest,
)


class EvalContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report = {
            "model_path": str(self.root / "local-model"),
            "checkpoint_signature": "fixture-fingerprint",
        }
        self.expected = make_manifest(self.report, DEFAULT_LAYERS)
        self.remote = {
            **self.expected,
            "model_path": "/server/models/dsv4",
            "runtime_quantization": {"method": None},
        }
        self.saved = {
            "target_hidden_state_format": HS_FORMAT,
            "aux_hidden_state_layer_ids": list(DEFAULT_LAYERS),
            "speculators_config": {
                "verifier": {"name_or_path": self.remote["model_path"]}
            },
            "target_training_contract": copy.deepcopy(self.remote),
        }

    def test_file_transport_ignores_model_path_without_rewriting_manifest(self):
        path = self.root / MANIFEST
        path.write_text(json.dumps(self.remote), encoding="utf-8")
        original = path.read_bytes()
        self.assertEqual(read_eval_manifest(self.root, self.expected), self.remote)
        self.assertEqual(path.read_bytes(), original)
        # Exporter/trainer validation must NOT adopt evaluation's relocation policy.
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ensure_manifest(self.root, self.expected)

    def test_only_model_path_and_runtime_override_are_ignored(self):
        for change in (
            {"checkpoint_signature": "wrong"},
            {"schema_version": 2},
            {"format": "standard"},
            {"auxiliary_hs_ids": [1, 2]},
            {"teacher_hs_id": 42},
            {"hidden_size": 128},
            {"target_dtype": "float16"},
            {"unexpected": True},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "mismatch"),
            ):
                validate_eval_manifest({**self.remote, **change}, self.expected)
        for field in self.expected.keys() - {"model_path"}:
            missing = dict(self.remote)
            del missing[field]
            with (
                self.subTest(missing=field),
                self.assertRaisesRegex(ValueError, "mismatch"),
            ):
                validate_eval_manifest(missing, self.expected)

    def test_missing_or_malformed_manifest_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            read_eval_manifest(self.root, self.expected)
        for value in (None, [], "invalid"):
            (self.root / MANIFEST).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid"):
                read_eval_manifest(self.root, self.expected)
        for path in (None, "", 42):
            with (
                self.subTest(path=path),
                self.assertRaisesRegex(ValueError, "model_path"),
            ):
                validate_eval_manifest(
                    {**self.remote, "model_path": path}, self.expected
                )

    def test_saved_contract_accepts_relocation_but_rejects_other_checkpoint(self):
        self.assertEqual(validate_draft_target(self.saved, self.report), DEFAULT_LAYERS)
        for model_path in (self.remote["model_path"], self.report["model_path"]):
            saved = copy.deepcopy(self.saved)
            saved["speculators_config"]["verifier"]["name_or_path"] = model_path
            saved["target_training_contract"]["model_path"] = model_path
            saved["target_training_contract"]["checkpoint_signature"] = "other"
            with (
                self.subTest(model_path=model_path),
                self.assertRaisesRegex(ValueError, "mismatch"),
            ):
                validate_draft_target(saved, self.report)

    def test_saved_contract_redundant_fields_remain_checked(self):
        for field, value in (
            ("target_hidden_state_format", "standard"),
            ("aux_hidden_state_layer_ids", [1, 2]),
            ("speculators_config", {"verifier": {"name_or_path": "/wrong"}}),
        ):
            saved = {**self.saved, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_draft_target(saved, self.report)

    def test_legacy_same_path_works_but_relocation_requires_saved_identity(self):
        del self.saved["target_training_contract"]
        with self.assertRaisesRegex(ValueError, "requires target_training_contract"):
            validate_draft_target(self.saved, self.report)
        self.saved["speculators_config"]["verifier"]["name_or_path"] = self.report[
            "model_path"
        ]
        self.assertEqual(validate_draft_target(self.saved, self.report), DEFAULT_LAYERS)

    def test_binding_changes_only_in_memory_loader_path_after_validation(self):
        original = copy.deepcopy(self.saved)
        config = SimpleNamespace(
            to_dict=lambda: copy.deepcopy(self.saved),
            target_training_contract=copy.deepcopy(
                self.saved["target_training_contract"]
            ),
            speculators_config=SimpleNamespace(
                verifier=SimpleNamespace(name_or_path=self.remote["model_path"])
            ),
        )
        bind_draft_verifier(config, self.report)
        self.assertEqual(
            config.speculators_config.verifier.name_or_path, self.report["model_path"]
        )
        self.assertEqual(self.saved, original)
        self.assertEqual(
            config.target_training_contract, original["target_training_contract"]
        )
        config.speculators_config.verifier.name_or_path = "do-not-change"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            bind_draft_verifier(
                config, {**self.report, "checkpoint_signature": "wrong"}
            )
        self.assertEqual(
            config.speculators_config.verifier.name_or_path, "do-not-change"
        )


if __name__ == "__main__":
    unittest.main()
