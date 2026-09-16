"""Run with PYTHONPATH=src python -m unittest discover -s tests/standalone."""

# ruff: noqa: PT009, PT027 -- Keep this suite runnable without pytest/torch.

import copy
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from speculators_dsv4 import HS_FORMAT
from speculators_dsv4.contract import (
    DEFAULT_LAYERS,
    MANIFEST,
    _read_header,
    ensure_manifest,
    inspect_checkpoint,
    make_manifest,
    replace_teacher_hidden,
    resolve_io_keys,
    validate_config,
    validate_layers,
)
from speculators_dsv4.training import prepare_training


def valid_config():
    return {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "hc_mult": 4,
        "vocab_size": 129280,
        "expert_dtype": "bf16",
        "torch_dtype": "bfloat16",
        "rms_norm_eps": 1e-6,
    }


def preview_config():
    return {
        **valid_config(),
        "expert_dtype": "fp4",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def write_tensor_file(path, tensors):
    """Tiny synthetic safetensors file; tensor contents are irrelevant to audit."""
    header = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        size = (
            math.prod(shape)
            * {
                "BF16": 2,
                "F16": 2,
                "F32": 4,
                "I8": 1,
                "U8": 1,
                "F8_E4M3": 1,
            }[dtype]
        )
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, extra=None, config=None):
        # Patch only geometry validation when inspecting tiny test files. The
        # strict real Flash geometry is independently tested below.
        cfg = {**(config or valid_config()), "hidden_size": 2, "vocab_size": 3}
        (self.root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        tensors = {
            "embed.weight": ("BF16", [3, 2]),
            "head.weight": ("BF16", [3, 2]),
            "norm.weight": ("F32", [2]),
            "mtp.norm.weight": ("F32", [2]),
        }
        tensors.update(extra or {})
        write_tensor_file(self.root / "model.safetensors", tensors)
        return tensors

    def inspect_tiny(self, *, require_bf16=False):
        with patch("speculators_dsv4.contract.validate_config"):
            return inspect_checkpoint(self.root, require_bf16=require_bf16)

    def test_config_accepts_original_quantized_checkpoint_geometry(self):
        validate_config(valid_config())
        validate_config(preview_config())
        for changed in (
            {"compression_config": {"format": "pack-quantized"}},
            {"torch_dtype": "float16"},
            {"quantization_config": {"quant_method": "ascend"}},
        ):
            with self.subTest(changed=changed):
                validate_config({**valid_config(), **changed})

    def test_config_still_rejects_unsupported_geometry(self):
        for changed in (
            {"hidden_size": 2560},
            {"hc_mult": 1},
            {"num_hidden_layers": 44},
            {"vocab_size": 151936},
            {"model_type": "qwen3"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_config({**valid_config(), **changed})

    def test_layer_ids_are_slots_with_separate_teacher(self):
        validate_layers(DEFAULT_LAYERS)
        validate_layers((1, 42))
        for bad in ([], [0, 10], [1, 43], [11, 1], [1, 1], [True], [1.0], ["1"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_layers(bad)

    def test_io_resolution_never_uses_mtp_norm(self):
        keys = {"embed.weight", "head.weight", "norm.weight", "mtp.norm.weight"}
        self.assertEqual(resolve_io_keys(keys)["model.norm.weight"], "norm.weight")
        for bad in (keys - {"norm.weight"}, keys | {"model.norm.weight"}):
            with self.assertRaises(ValueError):
                resolve_io_keys(bad)

    def test_checkpoint_headers_and_signature(self):
        self.fixture()
        report = self.inspect_tiny()
        self.assertEqual(report["tensor_count"], 4)
        self.assertEqual(report["shard_count"], 1)
        self.assertEqual(report["weight_dtypes"], {"BF16": 2, "F32": 2})
        self.assertEqual(report["quantization_files"], [])
        self.assertEqual(report, self.inspect_tiny(require_bf16=True))
        self.assertEqual(report, self.inspect_tiny())
        cfg_path = self.root / "config.json"
        cfg_path.write_text(cfg_path.read_text() + "\n", encoding="utf-8")
        self.assertNotEqual(
            report["checkpoint_signature"], self.inspect_tiny()["checkpoint_signature"]
        )

    def test_accepts_quantized_backbone_without_dequantizing_it(self):
        self.fixture(
            {
                "layers.0.ffn.experts.0.w1.weight": ("I8", [2, 2]),
                "layers.0.ffn.experts.0.w1.scale": ("U8", [2, 1]),
                "layers.0.attn.wq.weight": ("F8_E4M3", [2, 2]),
                "layers.0.attn.wq.weight_scale_inv": ("F32", [1]),
            },
            config=preview_config(),
        )
        report = self.inspect_tiny()
        self.assertEqual(
            report["weight_dtypes"],
            {"BF16": 2, "F32": 3, "F8_E4M3": 1, "I8": 1, "U8": 1},
        )
        self.assertEqual(report["config"]["expert_dtype"], "fp4")
        with self.assertRaisesRegex(ValueError, "quantization metadata remains"):
            self.inspect_tiny(require_bf16=True)

    def test_optional_strict_audit_rejects_packed_payload(self):
        self.fixture({"layers.0.ffn.experts.0.w1.weight": ("I8", [2, 2])})
        self.inspect_tiny()
        with self.assertRaisesRegex(ValueError, "Non-BF16"):
            self.inspect_tiny(require_bf16=True)

    def test_optional_strict_audit_rejects_unconsumed_float_scales(self):
        self.fixture({"layers.0.ffn.w1.scale": ("F32", [1])})
        self.inspect_tiny()
        with self.assertRaisesRegex(ValueError, "Unconsumed"):
            self.inspect_tiny(require_bf16=True)

    def test_optional_strict_audit_rejects_quantized_config(self):
        for changed in (
            {"expert_dtype": "fp4"},
            {"quantization_config": {"quant_method": "fp8"}},
            {"compression_config": {"format": "pack-quantized"}},
            {"torch_dtype": "float16"},
        ):
            with self.subTest(changed=changed):
                self.fixture(config={**valid_config(), **changed})
                self.inspect_tiny()
                with self.assertRaises(ValueError):
                    self.inspect_tiny(require_bf16=True)

    def test_plain_training_io_accepts_bf16_fp16_and_fp32(self):
        for dtype in ("BF16", "F16", "F32"):
            with self.subTest(dtype=dtype):
                self.fixture(
                    {
                        "embed.weight": (dtype, [3, 2]),
                        "head.weight": (dtype, [3, 2]),
                        "norm.weight": (dtype, [2]),
                    },
                    config=preview_config(),
                )
                self.inspect_tiny()

    def test_quantized_training_io_has_local_actionable_error(self):
        for key, shape in (
            ("embed.weight", [3, 2]),
            ("head.weight", [3, 2]),
            ("norm.weight", [2]),
        ):
            for dtype in ("F8_E4M3", "I8"):
                with self.subTest(key=key, dtype=dtype):
                    self.fixture({key: (dtype, shape)}, config=preview_config())
                    with self.assertRaisesRegex(
                        ValueError, "Quantized training IO"
                    ) as raised:
                        self.inspect_tiny()
                    self.assertIn(key, str(raised.exception))
                    self.assertIn(
                        "Dequantize only the affected training IO",
                        str(raised.exception),
                    )

    def test_training_io_scales_are_not_silently_ignored(self):
        for key in (
            "embed.weight_scale",
            "head.scale",
            "head.weight_scale_inv",
            "norm.input_scale",
        ):
            with self.subTest(key=key):
                self.fixture({key: ("F32", [1])})
                with self.assertRaisesRegex(ValueError, "Quantized training IO"):
                    self.inspect_tiny()

    def test_mtp_scales_do_not_block_unquantized_training_io(self):
        self.fixture({"mtp.norm.weight_scale": ("F32", [1])})
        self.inspect_tiny()

    def test_rejects_io_shape_mismatch(self):
        self.fixture({"head.weight": ("BF16", [2, 3])})
        with self.assertRaisesRegex(ValueError, "Wrong target IO shape"):
            self.inspect_tiny()

    def test_valid_index_and_missing_tensor(self):
        tensors = self.fixture()
        index = self.root / "model.safetensors.index.json"
        weight_map = dict.fromkeys(tensors, "model.safetensors")
        index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
        self.inspect_tiny()
        weight_map["missing.weight"] = "model.safetensors"
        index.write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing tensors"):
            self.inspect_tiny()

    def test_rejects_index_shard_outside_root(self):
        self.fixture()
        child = self.root / "checkpoint"
        child.mkdir()
        (child / "config.json").write_bytes((self.root / "config.json").read_bytes())
        (child / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"norm.weight": "../model.safetensors"}}),
            encoding="utf-8",
        )
        with (
            patch("speculators_dsv4.contract.validate_config"),
            self.assertRaisesRegex(ValueError, "escapes"),
        ):
            inspect_checkpoint(child)

    def test_rejects_truncated_or_overlapping_payload(self):
        path = self.root / "bad.safetensors"
        for payload in (b"", struct.pack("<Q", 50) + b"{}"):
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                _read_header(path)
        write_tensor_file(path, {"a": ("BF16", [2])})
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "payload"):
            _read_header(path)
        entry = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        raw = json.dumps({"a": entry, "b": entry}).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            _read_header(path)

    def test_manifest_read_only_validation_and_mismatch(self):
        report = {"model_path": str(self.root), "checkpoint_signature": "fixture"}
        expected = make_manifest(report, DEFAULT_LAYERS)
        hs_dir = self.root / "hs"
        with self.assertRaisesRegex(ValueError, "Missing"):
            ensure_manifest(hs_dir, expected)
        self.assertFalse(hs_dir.exists())
        ensure_manifest(hs_dir, expected, create=True)
        original = (hs_dir / MANIFEST).read_bytes()
        ensure_manifest(hs_dir, expected)
        changed = {**expected, "auxiliary_hs_ids": [1, 42]}
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ensure_manifest(hs_dir, changed, create=True)
        self.assertEqual((hs_dir / MANIFEST).read_bytes(), original)

    def test_manifest_does_not_relabel_old_hidden_states(self):
        hs_dir = self.root / "hs"
        hs_dir.mkdir()
        (hs_dir / "hs_0.safetensors").write_bytes(b"old")
        with self.assertRaisesRegex(ValueError, "existing HS"):
            ensure_manifest(hs_dir, {"format": HS_FORMAT}, create=True)
        self.assertFalse((hs_dir / MANIFEST).exists())

    def test_runtime_quantization_manifest_reuses_only_identical_mode(self):
        report = {"model_path": str(self.root), "checkpoint_signature": "fixture"}
        expected = make_manifest(report, DEFAULT_LAYERS)
        for method in (None, "ascend", "fp8"):
            with self.subTest(method=method):
                hs_dir = self.root / f"hs_{method}"
                runtime = {"method": method}
                ensure_manifest(
                    hs_dir, expected, create=True, runtime_quantization=runtime
                )
                original = (hs_dir / MANIFEST).read_bytes()
                saved = json.loads(original)
                self.assertEqual(saved["runtime_quantization"], runtime)
                self.assertNotIn("runtime_quantization", expected)
                ensure_manifest(hs_dir, expected, runtime_quantization=runtime)
                for changed in (None, "ascend", "fp8"):
                    if changed == method:
                        continue
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        ensure_manifest(
                            hs_dir,
                            expected,
                            create=True,
                            runtime_quantization={"method": changed},
                        )
                    self.assertEqual((hs_dir / MANIFEST).read_bytes(), original)

    def test_runtime_annotation_does_not_weaken_training_manifest_check(self):
        report = {"model_path": str(self.root), "checkpoint_signature": "fixture"}
        expected = make_manifest(report, DEFAULT_LAYERS)
        hs_dir = self.root / "hs"
        ensure_manifest(
            hs_dir,
            expected,
            create=True,
            runtime_quantization={"method": "ascend"},
        )
        ensure_manifest(hs_dir, expected)
        for changed in (
            {"checkpoint_signature": "another-checkpoint"},
            {"auxiliary_hs_ids": [1, 42]},
            {"target_dtype": "float16"},
            {"unexpected_field": "must-not-be-ignored"},
        ):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "mismatch"),
            ):
                ensure_manifest(hs_dir, {**expected, **changed})

    def test_old_manifest_is_not_rewritten_to_claim_runtime_quantization(self):
        report = {"model_path": str(self.root), "checkpoint_signature": "fixture"}
        expected = make_manifest(report, DEFAULT_LAYERS)
        hs_dir = self.root / "hs"
        ensure_manifest(hs_dir, expected, create=True)
        original = (hs_dir / MANIFEST).read_bytes()
        for method in (None, "ascend"):
            with (
                self.subTest(method=method),
                self.assertRaisesRegex(ValueError, "mismatch"),
            ):
                ensure_manifest(
                    hs_dir,
                    expected,
                    create=True,
                    runtime_quantization={"method": method},
                )
            self.assertEqual((hs_dir / MANIFEST).read_bytes(), original)

    def test_quantization_sidecar_change_invalidates_hidden_states(self):
        self.fixture(config=preview_config())
        original = self.inspect_tiny()
        hs_dir = self.root / "hs"
        ensure_manifest(hs_dir, make_manifest(original, DEFAULT_LAYERS), create=True)
        sidecar = self.root / "quant_model_description.json"
        sidecar.write_text('{"quant_type": "W8A8"}', encoding="utf-8")
        added = self.inspect_tiny()
        self.assertNotEqual(
            original["checkpoint_signature"], added["checkpoint_signature"]
        )
        self.assertEqual(added["quantization_files"], [sidecar.name])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ensure_manifest(hs_dir, make_manifest(added, DEFAULT_LAYERS))
        sidecar.write_text('{"quant_type": "W8A8_DYNAMIC"}', encoding="utf-8")
        self.assertNotEqual(
            added["checkpoint_signature"], self.inspect_tiny()["checkpoint_signature"]
        )

    def test_compression_sidecar_is_included_in_signature(self):
        self.fixture()
        original = self.inspect_tiny()
        sidecar = self.root / "compression_config.json"
        sidecar.write_text('{"format": "pack-quantized"}', encoding="utf-8")
        changed = self.inspect_tiny()
        self.assertEqual(changed["quantization_files"], [sidecar.name])
        self.assertNotEqual(
            original["checkpoint_signature"], changed["checkpoint_signature"]
        )

    def test_teacher_replaces_only_last_slot(self):
        normalized = object()
        auxiliary = [SimpleNamespace(shape=(5, 4096)) for _ in range(6)]
        teacher = SimpleNamespace(shape=(5, 4096), ndim=2)
        result, exported = replace_teacher_hidden((normalized, auxiliary), teacher, 6)
        self.assertIs(result, normalized)
        for i in range(5):
            self.assertIs(exported[i], auxiliary[i])
        self.assertIs(exported[-1], teacher)
        self.assertIsNot(auxiliary[-1], teacher)
        for output, captured, count in (
            (normalized, teacher, 6),
            ((normalized, auxiliary), None, 6),
            ((normalized, auxiliary), teacher, 5),
            ((normalized, auxiliary), SimpleNamespace(ndim=3), 6),
            ((normalized, auxiliary), SimpleNamespace(ndim=2, shape=(3, 4096)), 6),
        ):
            with self.assertRaises(RuntimeError):
                replace_teacher_hidden(output, captured, count)

    def test_training_requires_manifest_and_explicit_decoder(self):
        args = SimpleNamespace(
            speculator_type="dspark",
            hidden_states_backend="file",
            legacy_data=False,
            verifier_name_or_path=str(self.root),
            from_pretrained=None,
            draft_config="dense.json",
            target_layer_ids=None,
            mask_token_id=None,
            dry_run=True,
            hidden_states_path=str(self.root / "hs"),
            data_path="unused",
        )
        report = {
            "config": valid_config(),
            "model_path": str(self.root),
            "checkpoint_signature": "fixture",
        }
        with (
            patch("speculators_dsv4.training.inspect_checkpoint", return_value=report),
            patch("speculators_dsv4.training.validate_data_manifest"),
        ):
            with self.assertRaisesRegex(ValueError, "dry-run"):
                prepare_training(args)
            self.assertEqual(args.target_layer_ids, DEFAULT_LAYERS)
            self.assertEqual(args.mask_token_id, 128799)
            self.assertFalse(Path(args.hidden_states_path).exists())
            args.dry_run = False
            with self.assertRaisesRegex(ValueError, "Missing"):
                prepare_training(args)
            ensure_manifest(
                args.hidden_states_path,
                make_manifest(report, DEFAULT_LAYERS),
                create=True,
                runtime_quantization={"method": None},
            )
            prepare_training(args)
            self.assertEqual(
                args.target_training_contract["runtime_quantization"], {"method": None}
            )
            for changed in (
                {"draft_config": None},
                {"speculator_type": "dflash"},
                {"hidden_states_backend": "other"},
                {"legacy_data": True},
            ):
                invalid = copy.copy(args)
                invalid.__dict__.update(changed)
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    prepare_training(invalid)

    def test_training_uses_quantized_checkpoint_with_plain_io(self):
        self.fixture(
            {"layers.0.ffn.experts.0.w1.weight": ("I8", [2, 2])},
            config=preview_config(),
        )
        args = SimpleNamespace(
            speculator_type="dspark",
            hidden_states_backend="file",
            legacy_data=False,
            verifier_name_or_path=str(self.root),
            from_pretrained=None,
            draft_config="dense.json",
            target_layer_ids=None,
            mask_token_id=None,
            dry_run=False,
            hidden_states_path=str(self.root / "hs"),
            data_path="unused",
        )
        report = self.inspect_tiny()
        ensure_manifest(
            args.hidden_states_path,
            make_manifest(report, DEFAULT_LAYERS),
            create=True,
            runtime_quantization={"method": "fp8"},
        )
        with (
            patch("speculators_dsv4.contract.validate_config"),
            patch("speculators_dsv4.training.validate_data_manifest"),
        ):
            prepared = prepare_training(args)
        self.assertEqual(prepared, report)
        self.assertEqual(args.target_layer_ids, DEFAULT_LAYERS)
        self.assertEqual(
            prepared["config"]["quantization_config"]["quant_method"], "fp8"
        )


if __name__ == "__main__":
    unittest.main()
