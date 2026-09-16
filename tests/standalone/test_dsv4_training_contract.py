"""Training identity tests runnable with only the Python standard library."""

# ruff: noqa: PT009, PT027 -- Also usable without pytest/torch.

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import DEFAULT_LAYERS, MANIFEST, make_manifest
from speculators_dsv4.training import prepare_training
from speculators_dsv4.training_contract import (
    distributed_validation,
    read_training_contract,
    validate_draft_contract,
    validate_resume_contract,
    validate_training_contract,
)


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.report = {
            "model_path": str(self.root / "target"),
            "checkpoint_signature": "a" * 64,
            "config": {"dspark_noise_token_id": 128799},
        }
        self.contract = make_manifest(self.report, DEFAULT_LAYERS)
        self.contract["runtime_quantization"] = {"method": "ascend"}
        self.hs = self.root / "hs"
        self.hs.mkdir()
        self.write_manifest(self.contract)
        self.saved = {
            "target_hidden_state_format": HS_FORMAT,
            "target_training_contract": copy.deepcopy(self.contract),
            "aux_hidden_state_layer_ids": list(DEFAULT_LAYERS),
            "speculators_config": {
                "verifier": {"name_or_path": self.report["model_path"]}
            },
        }

    def write_manifest(self, manifest):
        (self.hs / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    def args(self, **changes):
        args = SimpleNamespace(
            speculator_type="dspark",
            hidden_states_backend="file",
            legacy_data=False,
            verifier_name_or_path=self.report["model_path"],
            from_pretrained=None,
            draft_config="dense.json",
            target_layer_ids=None,
            mask_token_id=None,
            hidden_states_path=str(self.hs),
            data_path="unused",
            dry_run=False,
        )
        vars(args).update(changes)
        return args

    def prepare(self, args, saved=None):
        fake = SimpleNamespace(
            PretrainedConfig=SimpleNamespace(
                get_config_dict=Mock(return_value=(saved, {}))
            )
        )
        with (
            patch(
                "speculators_dsv4.training.inspect_checkpoint", return_value=self.report
            ),
            patch.dict("sys.modules", {"transformers": fake}),
            patch("speculators_dsv4.training.validate_data_manifest") as data_check,
        ):
            result = prepare_training(args)
        data_check.assert_called_once_with(args.data_path, self.report)
        return result

    def test_foreign_or_unproven_data_rejected_before_checkpoint_build(self):
        args = self.args()
        with (
            patch(
                "speculators_dsv4.training.inspect_checkpoint", return_value=self.report
            ),
            patch(
                "speculators_dsv4.training.validate_data_manifest",
                side_effect=ValueError("DSV4 training data identity mismatch"),
            ),
            self.assertRaisesRegex(ValueError, "training data identity mismatch"),
        ):
            prepare_training(args)
        self.assertFalse(hasattr(args, "target_training_contract"))

    def test_fresh_training_binds_independent_manifest_copy(self):
        args = self.args()
        self.assertEqual(self.prepare(args), self.report)
        self.assertEqual(args.target_training_contract, self.contract)
        args.target_training_contract["auxiliary_hs_ids"].append(42)
        self.assertEqual(json.loads((self.hs / MANIFEST).read_text()), self.contract)

    def test_dry_run_requires_real_manifest_and_records_it(self):
        args = self.args(dry_run=True)
        self.prepare(args)
        self.assertEqual(args.target_training_contract, self.contract)
        args = self.args(dry_run=True, hidden_states_path=str(self.root / "missing"))
        with self.assertRaisesRegex(ValueError, "dry-run"):
            self.prepare(args)
        self.assertFalse(hasattr(args, "target_training_contract"))
        self.assertFalse((self.root / "missing").exists())

    def test_legacy_manifest_not_labeled_with_current_runtime(self):
        legacy = make_manifest(self.report, DEFAULT_LAYERS)
        self.write_manifest(legacy)
        with self.assertRaisesRegex(ValueError, "runtime_quantization"):
            self.prepare(self.args())
        self.assertEqual(json.loads((self.hs / MANIFEST).read_text()), legacy)

    def test_none_runtime_method_is_explicit_and_distinct_from_missing(self):
        self.contract["runtime_quantization"] = {"method": None}
        self.write_manifest(self.contract)
        self.assertEqual(
            read_training_contract(self.hs, self.report, DEFAULT_LAYERS), self.contract
        )

    def test_manifest_rejects_changed_checkpoint_signature(self):
        report = dict(self.report, checkpoint_signature="b" * 64)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            read_training_contract(self.hs, report, DEFAULT_LAYERS)

    def test_fixed_schema_and_field_types(self):
        variants = [
            {"schema_version": True},
            {"schema_version": 2},
            {"format": "standard"},
            {"teacher_hs_id": 42},
            {"target_dtype": "float16"},
            {"hidden_size": 1},
            {"model_path": ""},
            {"checkpoint_signature": None},
            {"auxiliary_hs_ids": (1, 11)},
            {"auxiliary_hs_ids": [True]},
            {"runtime_quantization": None},
            {"runtime_quantization": {}},
            {"runtime_quantization": {"method": 1}},
            {"runtime_quantization": {"method": ""}},
            {"runtime_quantization": {"method": "ascend", "extra": 1}},
            {"unexpected": "field"},
        ]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                validate_training_contract({**self.contract, **variant})

    def test_valid_pretrained_preserves_recorded_identity_and_layers(self):
        args = self.args(from_pretrained="saved-draft")
        self.prepare(args, self.saved)
        self.assertEqual(args.target_training_contract, self.contract)
        self.assertEqual(args.target_layer_ids, DEFAULT_LAYERS)

    def test_old_pretrained_without_identity_rejected(self):
        del self.saved["target_training_contract"]
        with self.assertRaisesRegex(ValueError, "identity cannot be proven"):
            self.prepare(self.args(from_pretrained="old-draft"), self.saved)

    def test_pretrained_same_path_different_checkpoint_rejected(self):
        self.saved["target_training_contract"]["checkpoint_signature"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "checkpoint_signature"):
            self.prepare(self.args(from_pretrained="saved-draft"), self.saved)

    def test_pretrained_new_hs_directory_changed_runtime_rejected(self):
        self.contract["runtime_quantization"] = {"method": "fp8"}
        self.write_manifest(self.contract)
        with self.assertRaisesRegex(ValueError, "runtime_quantization"):
            self.prepare(self.args(from_pretrained="saved-draft"), self.saved)

    def test_pretrained_cannot_change_cli_layers(self):
        with self.assertRaisesRegex(ValueError, "Cannot change"):
            self.prepare(
                self.args(from_pretrained="saved-draft", target_layer_ids=[1, 12]),
                self.saved,
            )

    def test_saved_config_redundant_identity_fields_checked(self):
        for field, value, message in (
            ("target_hidden_state_format", "standard", "HS format"),
            ("aux_hidden_state_layer_ids", [1, 12], "HS layers"),
            ("speculators_config", {"verifier": {"name_or_path": "elsewhere"}}, "path"),
        ):
            saved = {**self.saved, field: value}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                validate_draft_contract(saved, self.contract, source="fixture")

    def test_auto_resume_uses_saved_config(self):
        checkpoint = self.root / "run" / "0"
        checkpoint.mkdir(parents=True)
        path = checkpoint / "config.json"
        path.write_text(json.dumps(self.saved), encoding="utf-8")
        validate_resume_contract(self.saved, checkpoint)
        stale = copy.deepcopy(self.saved)
        stale["target_training_contract"]["runtime_quantization"] = {"method": None}
        path.write_text(json.dumps(stale), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "runtime_quantization"):
            validate_resume_contract(self.saved, checkpoint)

    def test_auto_resume_missing_or_legacy_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing DSV4 resume configuration"):
            validate_resume_contract(self.saved, self.root)
        legacy = copy.deepcopy(self.saved)
        del legacy["target_training_contract"]
        (self.root / "config.json").write_text(json.dumps(legacy), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity cannot be proven"):
            validate_resume_contract(self.saved, self.root)

    def test_new_training_checks_bound_live_config_without_resume(self):
        validate_resume_contract(self.saved, None)
        self.saved["target_hidden_state_format"] = "standard"
        with self.assertRaisesRegex(ValueError, "HS format"):
            validate_resume_contract(self.saved, None)


class DistributedValidationTests(unittest.TestCase):
    def distributed(self, remote_error=None):
        def gather(errors, error):
            errors[:] = [error, remote_error]

        return SimpleNamespace(
            get_world_size=Mock(return_value=2),
            all_gather_object=Mock(side_effect=gather),
        )

    def test_local_success_returns_value(self):
        self.assertEqual(distributed_validation(lambda: 123), 123)

    def test_local_error_keeps_original_exception(self):
        with self.assertRaisesRegex(OSError, "unreadable"):
            distributed_validation(Mock(side_effect=OSError("unreadable")))

    def test_all_ranks_exchange_success_before_return(self):
        distributed = self.distributed()
        self.assertEqual(distributed_validation(lambda: 123, distributed), 123)
        distributed.all_gather_object.assert_called_once()

    def test_local_failure_is_exchanged_before_raising(self):
        distributed = self.distributed()
        with self.assertRaisesRegex(ValueError, "rank 0: OSError: unreadable"):
            distributed_validation(Mock(side_effect=OSError("unreadable")), distributed)
        distributed.all_gather_object.assert_called_once()

    def test_other_rank_failure_stops_local_rank(self):
        with self.assertRaisesRegex(ValueError, "rank 1: ValueError: stale checkpoint"):
            distributed_validation(
                lambda: 123, self.distributed("ValueError: stale checkpoint")
            )


if __name__ == "__main__":
    unittest.main()
