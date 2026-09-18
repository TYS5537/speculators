"""Read-only checkpoint checks and the BF16 exported hidden-state contract.

Backbone quantization belongs to the target backend. Only the training-side
embedding, head and norm must be ordinary floating-point tensors. Header checks
do NOT validate quantization kernels or numerical parity with target logits.
"""

import hashlib
import json
import math
import os
import struct
import time
import uuid
from pathlib import Path

from speculators_dsv4 import HS_FORMAT

DEFAULT_LAYERS = [1, 11, 21, 30, 40]
MANIFEST = "dspark_dsv4_hs.json"
_DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
_MAX_HEADER_BYTES = 100_000_000
_OUTPUT_PARTS = 2
IO_ALIASES = {
    "embed_tokens.weight": (
        "embed.weight",
        "model.embed.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ),
    "lm_head.weight": ("head.weight", "model.head.weight", "lm_head.weight"),
    "model.norm.weight": ("norm.weight", "model.norm.weight"),
}


def validate_config(config):
    """Validate model geometry, independently of stored weight precision."""
    expected = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "hc_mult": 4,
        "vocab_size": 129280,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"DSV4-Flash requires {key}={value}, got {config.get(key)}"
            )


def _validate_bf16_config(config):
    """Optional strict audit for an explicitly dequantized checkpoint."""
    if config.get("quantization_config") or config.get("compression_config"):
        raise ValueError(
            "Convert the checkpoint to BF16 first; quantization metadata remains."
        )
    if config.get("expert_dtype") not in {"bf16", "bfloat16"}:
        raise ValueError("Converted checkpoint must declare expert_dtype=bf16.")
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in {"bf16", "bfloat16"}:
        raise ValueError(
            "Converted checkpoint must declare dtype/torch_dtype=bfloat16."
        )


def validate_layers(layers, num_layers=43):
    """IDs are HS slots: i means decoder block i-1, not zero-based block i."""
    if any(type(i) is not int or not 1 <= i < num_layers for i in layers):
        raise ValueError(
            f"Use auxiliary HS IDs in [1, {num_layers - 1}]; "
            "teacher is appended separately."
        )
    if not layers or list(layers) != sorted(set(layers)):
        raise ValueError(
            "DSV4 auxiliary HS IDs must be nonempty, unique and ascending."
        )


def resolve_io_keys(keys):
    result = {}
    for logical, aliases in IO_ALIASES.items():
        found = [key for key in aliases if key in keys]
        if len(found) != 1:
            raise ValueError(
                f"Expected exactly one target tensor for {logical}; found {found}"
            )
        result[logical] = found[0]
    return result


def _read_header(path):  # noqa: C901
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != struct.calcsize("<Q"):
            raise ValueError(f"Truncated safetensors header: {path}")
        length = struct.unpack("<Q", length_bytes)[0]
        if not 0 < length <= _MAX_HEADER_BYTES:
            raise ValueError(f"Invalid safetensors header length: {path}")
        raw = stream.read(length)
        if len(raw) != length:
            raise ValueError(f"Truncated safetensors header: {path}")
    header = json.loads(raw)
    data_size = path.stat().st_size - 8 - length
    ranges = []
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= data_size
        ):
            raise ValueError(f"Truncated/invalid tensor payload: {path}: {name}")
        ranges.append((start, end))
        shape = entry["shape"]
        if any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"Invalid tensor shape: {path}: {name}")
        element_size = _DTYPE_BYTES.get(entry["dtype"])
        if element_size is not None and end - start != math.prod(shape) * element_size:
            raise ValueError(f"Tensor shape/payload size mismatch: {path}: {name}")
    offset = 0
    for start, end in sorted(ranges):
        if start != offset:
            raise ValueError(f"Overlapping or non-contiguous tensor payload: {path}")
        offset = end
    if offset != data_size:
        raise ValueError(f"Unindexed tensor payload: {path}")
    return {k: v for k, v in header.items() if k != "__metadata__"}


def inspect_checkpoint(model_path, *, require_bf16=False):  # noqa: C901
    """Audit local headers and training IO without dequantizing the backbone.

    ``require_bf16`` is an optional conversion audit, not an HS-export requirement.
    Actual target quantization support is checked by the inference backend.
    """
    root = Path(model_path).resolve(strict=True)
    config_bytes = (root / "config.json").read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    if require_bf16:
        _validate_bf16_config(config)
    index_path = root / "model.safetensors.index.json"
    weight_map = None
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shard_names = sorted(set(weight_map.values()))
    else:
        shard_names = ["model.safetensors"]
    if not shard_names:
        raise ValueError("Empty checkpoint weight map.")
    signature = hashlib.sha256(config_bytes)
    # Some backends keep their quantization recipe outside config.json. Include
    # those sidecars so changing them cannot silently reuse an old HS directory.
    quantization_files = []
    for metadata in sorted(root.glob("*.json")):
        if (
            "quant" not in metadata.name.lower()
            and "compress" not in metadata.name.lower()
        ):
            continue
        if not metadata.resolve(strict=True).is_relative_to(root):
            raise ValueError(f"Quantization metadata escapes checkpoint: {metadata}")
        signature.update(metadata.name.encode())
        signature.update(metadata.read_bytes())
        quantization_files.append(metadata.name)
    tensors = {}
    weight_dtypes = {}
    for filename in shard_names:
        path = (root / filename).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Shard escapes checkpoint directory: {filename}")
        header = _read_header(path)
        stat = path.stat()
        signature.update(
            json.dumps(
                [filename, stat.st_size, stat.st_mtime_ns, header],
                sort_keys=True,
            ).encode()
        )
        for name, entry in header.items():
            if name in tensors:
                raise ValueError(f"Duplicate tensor: {name}")
            if weight_map is not None and weight_map.get(name) != filename:
                raise ValueError(f"Index/header mismatch for {name}")
            dtype = entry["dtype"]
            weight_dtypes[dtype] = weight_dtypes.get(dtype, 0) + 1
            if require_bf16 and dtype not in {"BF16", "F32", "I64", "I32", "BOOL"}:
                raise ValueError(
                    f"Non-BF16 checkpoint tensor {name}: {dtype}; "
                    "dequantize with scales first."
                )
            if (
                require_bf16
                and name.endswith(".weight")
                and dtype not in {"BF16", "F32"}
            ):
                raise ValueError(f"Packed/non-floating weight {name}: {dtype}")
            if require_bf16 and (
                "scale_inv" in name or name.endswith((".weight_scale", ".scale"))
            ):
                raise ValueError(f"Unconsumed quantization scale: {name}")
            tensors[name] = entry
    if weight_map is not None and set(weight_map) != set(tensors):
        raise ValueError("Checkpoint index references missing tensors.")
    io_keys = resolve_io_keys(tensors)
    for logical, actual in io_keys.items():
        # The trainer copies these three tensors into ordinary frozen modules;
        # loading a packed tensor or ignoring its scales would change the teacher.
        prefix = actual.removesuffix(".weight") + "."
        scales = [
            name for name in tensors if name.startswith(prefix) and "scale" in name
        ]
        if tensors[actual]["dtype"] not in {"BF16", "F16", "F32"} or scales:
            raise ValueError(
                f"Quantized training IO {actual}: dtype={tensors[actual]['dtype']}, "
                f"scales={scales}. The trainer needs unquantized floating-point "
                "embedding/head/norm. Dequantize only the affected training IO "
                "with its scales and verify teacher-logit parity; the target "
                "backbone does not need full BF16 conversion."
            )
        expected = (
            [config["hidden_size"]]
            if logical == "model.norm.weight"
            else [config["vocab_size"], config["hidden_size"]]
        )
        if tensors[actual]["shape"] != expected:
            raise ValueError(
                f"Wrong target IO shape for {actual}: {tensors[actual]['shape']}"
            )
    return {
        "model_path": str(root),
        "config": config,
        "io_keys": io_keys,
        "checkpoint_signature": signature.hexdigest(),
        "tensor_count": len(tensors),
        "shard_count": len(shard_names),
        "weight_dtypes": dict(sorted(weight_dtypes.items())),
        "quantization_files": quantization_files,
    }


def make_manifest(report, layers):
    validate_layers(layers)
    return {
        "schema_version": 1,
        "format": HS_FORMAT,
        "model_path": report["model_path"],
        "checkpoint_signature": report["checkpoint_signature"],
        "auxiliary_hs_ids": list(layers),
        "teacher_hs_id": 43,
        "hidden_size": 4096,
        "target_dtype": "bfloat16",  # Runtime/HS dtype, not stored weight precision.
    }


def ensure_manifest(directory, expected, *, create=False, runtime_quantization=None):
    # The server knows its explicit quantization override and must reject reuse
    # after a change. Consumers need not repeat that CLI flag: they still verify
    # the complete checkpoint/HS contract while reading the server's activations.
    expected = dict(expected)
    if runtime_quantization is not None:
        expected["runtime_quantization"] = runtime_quantization
    directory = Path(directory)
    path = directory / MANIFEST
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if runtime_quantization is None:
            actual.pop("runtime_quantization", None)
        if actual != expected:
            raise ValueError(
                f"DSV4 HS contract/target mismatch: {path}. Use a fresh HS directory."
            )
        return
    if not create:
        raise ValueError(f"Missing {path}; start the DSV4 HS server first.")
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(
            "Refusing to label existing HS files as DSV4. Use a new empty directory."
        )
    directory.mkdir(parents=True, exist_ok=True)
    # A second host may already be waiting for this file. Publish complete JSON
    # atomically without replacing an existing contract (also on shared NFS).
    temporary = None
    try:
        candidate = directory / f".{MANIFEST}.{uuid.uuid4().hex}.tmp"
        # Keep normal umask-controlled permissions, like the old direct write;
        # NamedTemporaryFile's 0600 mode would block other target/trainer users.
        with candidate.open("x", encoding="utf-8") as stream:
            temporary = candidate
            json.dump(expected, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text(encoding="utf-8")) != expected:
                raise ValueError(
                    f"DSV4 HS contract/target mismatch: {path}. "
                    "Use a fresh HS directory."
                ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def wait_for_manifest(directory, expected, *, runtime_quantization, timeout=300):
    """Secondary target nodes only read the head node's published contract.

    This validates matching metadata, not cross-host filesystem identity or NPU
    readiness. Shared-file/lock visibility still needs an actual HS probe.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("DSV4 manifest timeout must be finite and positive.")
    path = Path(directory) / MANIFEST
    deadline = time.monotonic() + timeout
    while not path.exists():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for DSV4 HS manifest: {path}. Start the target "
                "head node and mount the same shared HS directory at the same "
                "absolute path on all target/trainer hosts."
            )
        time.sleep(min(1, remaining))
    ensure_manifest(directory, expected, runtime_quantization=runtime_quantization)


def replace_teacher_hidden(output, teacher, count):
    """Retain backend-produced means; replace ONLY the final teacher slot."""
    if not isinstance(output, tuple) or len(output) != _OUTPUT_PARTS or teacher is None:
        raise RuntimeError(
            "DSV4 export did not capture post-hc_head/pre-norm hidden states."
        )
    normalized, auxiliary = output
    if (
        count is None
        or count < _OUTPUT_PARTS
        or len(auxiliary) != count
        or teacher.ndim != _OUTPUT_PARTS
    ):
        raise RuntimeError("Unexpected DSV4 auxiliary HS layout.")
    if any(h.shape != teacher.shape for h in auxiliary):
        raise RuntimeError(
            "DSV4 token/hidden dimensions differ (SP/CP export is unsupported)."
        )
    return normalized, [*auxiliary[:-1], teacher]
