"""Qwen3 BF16 hidden states over the existing authenticated HTTP transport.

The target's ordinary vLLM file connector writes local safetensors; a sidecar
serves only owned training requests. Checkpoint bytes and ordered layer IDs bind
the two hosts even when their model directories have different names.
"""

import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from uuid import uuid4

import torch

from speculators_dsv4.hs_http import MAX_MANIFEST_BYTES, validate_token
from speculators_dsv4.hs_http_server import HiddenStatesServer, _regular_file
from speculators_dsv4.hs_training_http import HttpTrainingTransfer

MANIFEST = "qwen3_hs_http.json"
LOGGER = logging.getLogger(__name__)
_HIDDEN_NDIM = 3
_MAX_PORT = 65535
_FORMAT = "qwen3_standard_hidden_states_v1"
_FIELDS = {
    "format",
    "schema_version",
    "checkpoint_sha256",
    "auxiliary_hs_ids",
    "teacher_hs_id",
    "hidden_size",
    "vocab_size",
    "target_dtype",
}


def validate_contract(value):
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Invalid Qwen HTTP hidden-state contract")
    for name, expected in {
        "format": _FORMAT,
        "schema_version": 1,
        "target_dtype": "bfloat16",
    }.items():
        if type(value[name]) is not type(expected) or value[name] != expected:
            raise ValueError(f"Invalid Qwen HTTP contract {name}")
    for name in ("teacher_hs_id", "hidden_size", "vocab_size"):
        if type(value[name]) is not int or value[name] <= 0:
            raise ValueError(f"Invalid Qwen HTTP contract {name}")
    layers = value["auxiliary_hs_ids"]
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            type(i) is not int or not 0 <= i < value["teacher_hs_id"] for i in layers
        )
        or len(set(layers)) != len(layers)
    ):
        raise ValueError(
            "Qwen auxiliary layer IDs must be distinct and precede the teacher"
        )
    if not isinstance(value["checkpoint_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", value["checkpoint_sha256"]
    ):
        raise ValueError("Invalid Qwen checkpoint SHA256")
    return json.loads(json.dumps(value))


def checkpoint_contract(model_path, layers):
    """Hash local config/index and all safetensor shards, independently of paths."""
    root = Path(model_path).resolve(strict=True)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError(
            "Standard HTTP training currently supports local Qwen3 checkpoints"
        )
    if config.get("quantization_config") or config.get("compression_config"):
        raise ValueError("Qwen HTTP training requires an unquantized BF16 target")
    index = root / "model.safetensors.index.json"
    names = {"config.json"}
    if index.exists():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Empty or invalid checkpoint weight map")
        names.add(index.name)
        names.update(weight_map.values())
    else:
        names.add("model.safetensors")
    signature = hashlib.sha256()
    for name in sorted(names):
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Checkpoint shard escapes the model directory")
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Checkpoint changed while computing its identity")
        signature.update(name.encode() + b"\0" + digest.digest())
    return validate_contract(
        {
            "format": _FORMAT,
            "schema_version": 1,
            "checkpoint_sha256": signature.hexdigest(),
            "auxiliary_hs_ids": list(layers or []),
            "teacher_hs_id": config["num_hidden_layers"],
            "hidden_size": config["hidden_size"],
            "vocab_size": config["vocab_size"],
            "target_dtype": "bfloat16",
        }
    )


def save_contract(directory, contract):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST
    content = json.dumps(validate_contract(contract), sort_keys=True, indent=2) + "\n"
    if path.exists():
        if validate_contract(json.loads(path.read_text(encoding="utf-8"))) != contract:
            raise ValueError(
                "HS directory belongs to another model/layer selection; "
                "use a new directory"
            )
        return
    temporary = directory / (MANIFEST + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class QwenHttpTrainingTransfer(HttpTrainingTransfer):
    _validate_contract = staticmethod(validate_contract)

    def __init__(self, endpoints, token, directory, **kwargs):
        try:
            validate_token(token)
        except ValueError:
            raise ValueError(
                "Set SPECULATORS_HS_HTTP_TOKEN to at least 32 "
                "printable ASCII characters"
            ) from None
        if kwargs.get("expected_contract") is None:
            raise ValueError(
                "Qwen HTTP transfer requires a verified local checkpoint contract"
            )
        super().__init__(endpoints, token, directory, **kwargs)

    @staticmethod
    def _check_contract(actual, expected):
        if validate_contract(actual) != validate_contract(expected):
            raise ValueError("Qwen HTTP checkpoint/layer/dtype contract mismatch")

    def setup(self):
        super().setup()
        save_contract(self.directory, self._contract)

    def get_generated(self, handle):
        sample = super().get_generated(handle)
        hidden, tokens = sample.get("hidden_states"), sample.get("token_ids")
        contract = self._contract
        if (
            hidden is None
            or tokens is None
            or hidden.ndim != _HIDDEN_NDIM
            or tokens.ndim != 1
            or hidden.shape
            != (
                tokens.numel(),
                len(contract["auxiliary_hs_ids"]) + 1,
                contract["hidden_size"],
            )
            or hidden.dtype != torch.bfloat16
            or tokens.dtype not in (torch.int32, torch.int64)
        ):
            self._completed.pop(handle, None)
            raise ValueError(
                "Qwen HTTP hidden-state shape/dtype does not match its contract"
            )
        return sample


class QwenHiddenStatesServer(HiddenStatesServer):
    def read_manifest(self):
        with _regular_file(self.directory / MANIFEST, MAX_MANIFEST_BYTES) as stream:
            return validate_contract(json.load(stream))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "check"):
        command = sub.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--target-layer-ids", type=int, nargs="+", required=True)
        command.add_argument("--hidden-states-path", type=Path, required=True)
        if name == "check":
            command.add_argument("--hs-http-endpoints", nargs="+", required=True)
            command.add_argument("--timeout", type=float, default=120.0)
    server = sub.add_parser("serve")
    server.add_argument("--hidden-states-path", type=Path, required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()
    if args.command == "prepare":
        save_contract(
            args.hidden_states_path,
            checkpoint_contract(args.model, args.target_layer_ids),
        )
        LOGGER.info(
            "Prepared Qwen HTTP identity (full checkpoint SHA256 and ordered layers).",
        )
    elif args.command == "check":
        transfer = QwenHttpTrainingTransfer(
            args.hs_http_endpoints,
            os.environ.get("SPECULATORS_HS_HTTP_TOKEN"),
            args.hidden_states_path,
            timeout=args.timeout,
            expected_contract=checkpoint_contract(args.model, args.target_layer_ids),
        )
        transfer.setup()
        LOGGER.info(
            "Qwen HTTP target identity verified; hidden states will transfer over HTTP."
        )
    else:
        if not 1 <= args.port <= _MAX_PORT:
            parser.error("Port must be between 1 and 65535")
        with QwenHiddenStatesServer(
            (args.host, args.port),
            args.hidden_states_path,
            os.environ.get("SPECULATORS_HS_HTTP_TOKEN"),
            training=True,
        ) as instance:
            LOGGER.info(
                "Qwen hidden-state HTTP service on %s:%s",
                args.host,
                instance.server_port,
            )
            instance.serve_forever()


if __name__ == "__main__":
    main()
