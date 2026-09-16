import argparse
import json
import os
import sys
import warnings

try:
    from hs_connectors import HiddenStatesBackend

    _backend_registry: dict[str, type[HiddenStatesBackend]] = dict(
        HiddenStatesBackend.registry  # type: ignore[misc]
    )
except ImportError:
    _backend_registry = {}  # type: ignore[assignment]


if "file" not in _backend_registry:
    # Vendored File backend in case hs_connectors is not available
    class _InlineFileBackend:
        @staticmethod
        def add_launch_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument(
                "--hidden-states-path",
                type=str,
                default="/tmp/hidden_states",  # noqa: S108
                help=(
                    "The directory to save hidden states to. "
                    "Default '/tmp/hidden_states'"
                ),
            )

        @staticmethod
        def build_kv_transfer_config(args: argparse.Namespace) -> dict:
            return {
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": args.hidden_states_path,
                },
            }

    _backend_registry["file"] = _InlineFileBackend  # type: ignore[assignment]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for hidden states extraction",
        usage=(
            "launch_vllm.py [-h] MODEL [--hidden-states-backend BACKEND] "
            "[--target-layer-ids TARGET_LAYER_IDS [TARGET_LAYER_IDS ...]] -- *VLLM_ARGS"
        ),
    )
    parser.add_argument(
        "model", type=str, help="Model name or path to extract hidden states from"
    )

    parser.add_argument(
        "--hidden-states-backend",
        choices=list(_backend_registry.keys()),
        default="file",
        help=(
            "Hidden states transfer backend. Each backend may add its own "
            "CLI arguments (see below). Default: 'file'."
        ),
    )
    for backend_cls in _backend_registry.values():
        backend_cls.add_launch_args(parser)

    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) A (space separated) list of integer layer ids. Defaults to "
            "[2, num_hidden_layers // 2, num_hidden_layers - 3]. "
            "Note: if set, you must also pass the same value into the training process"
        ),
    )
    parser.add_argument(
        "--include-last-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the last layer (num_hidden_layers) to "
            "target_layer_ids for verifier hidden states extraction. Default: True"
        ),
    )
    parser.add_argument(
        "--dsv4",
        "--dsv4-bf16",
        dest="dsv4",
        action="store_true",
        help=(
            "Experimental DSV4-Flash BF16 HS bridge for vLLM Ascend 0.26.0rc1. "
            "Target weight quantization is handled by the backend; "
            "--dsv4-bf16 is a compatibility alias."
        ),
    )
    parser.add_argument(
        "--dsv4-block-verify",
        action="store_true",
        help=(
            "Dedicated DSV4 offline block-verification service; requires --dsv4. "
            "Only one sequence and max_tokens=1 requests are supported."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    return parser.parse_known_args()


def main():  # noqa: C901
    args, vllm_args = parse_args()
    if "--" in vllm_args:
        vllm_args.remove("--")
    if args.dsv4_block_verify and not args.dsv4:
        raise ValueError("--dsv4-block-verify requires --dsv4.")

    dsv4_manifest = None
    dsv4_runtime_quantization = None
    if args.dsv4:
        from importlib.metadata import entry_points  # noqa: PLC0415

        from speculators_dsv4 import ARCHITECTURE  # noqa: PLC0415
        from speculators_dsv4.contract import (  # noqa: PLC0415
            DEFAULT_LAYERS,
            ensure_manifest,
            inspect_checkpoint,
            make_manifest,
            validate_layers,
        )

        report = inspect_checkpoint(args.model)
        num_hidden_layers = report["config"]["num_hidden_layers"]
        if args.hidden_states_backend != "file" or not args.include_last_layer:
            raise ValueError("DSV4 requires the file backend and --include-last-layer.")
        args.target_layer_ids = args.target_layer_ids or list(DEFAULT_LAYERS)
        validate_layers(args.target_layer_ids)
        dsv4_manifest = make_manifest(report, args.target_layer_ids)
        quantization_parser = argparse.ArgumentParser(
            add_help=False, allow_abbrev=False
        )
        quantization_parser.add_argument("--quantization", "-q")
        quantization_args, _ = quantization_parser.parse_known_args(vllm_args)
        dsv4_runtime_quantization = {"method": quantization_args.quantization}
        if args.dsv4_block_verify:
            sequence_parser = argparse.ArgumentParser(
                add_help=False, allow_abbrev=False
            )
            sequence_parser.add_argument(
                "--max-num-seqs", "--max_num_seqs", type=int, action="append"
            )
            sequence_args, _ = sequence_parser.parse_known_args(vllm_args)
            if any(value != 1 for value in sequence_args.max_num_seqs or []):
                raise ValueError("DSV4 block verification requires --max-num-seqs 1.")
            if sequence_args.max_num_seqs is None:
                vllm_args.extend(["--max-num-seqs", "1"])
        for arg in vllm_args:
            if arg.split("=", 1)[0] in {
                "--hf-overrides",
                "--dtype",
                "--speculative-config",
                "--speculative_config",
                "--kv-transfer-config",
                "--kv_transfer_config",
            }:
                raise ValueError(
                    "DSV4 HS bridge owns HF overrides and runtime dtype, "
                    "speculative config and KV transfer config."
                )
        plugins = entry_points(group="vllm.general_plugins")
        if not args.dry_run and not any(p.name == "speculators_dsv4" for p in plugins):
            raise RuntimeError(
                "Install this checkout in the vLLM environment: "
                "pip install -e . --no-deps"
            )
        allowlist = os.environ.get("VLLM_PLUGINS")
        if allowlist is not None and "speculators_dsv4" not in allowlist.split(","):
            raise ValueError(
                "Add speculators_dsv4 to VLLM_PLUGINS (preserve your Ascend plugins)."
            )
        vllm_args.extend(
            [
                "--hf-overrides",
                json.dumps({"architectures": [ARCHITECTURE]}),
                "--dtype",
                "bfloat16",
                "--enforce-eager",
                "--no-enable-prefix-caching",
            ]
        )
    else:
        from transformers import AutoConfig  # noqa: PLC0415

        config = AutoConfig.from_pretrained(args.model)
        if hasattr(config, "text_config"):
            config = config.text_config
        num_hidden_layers = config.num_hidden_layers
        if getattr(config, "model_type", None) == "deepseek_v4":
            raise ValueError(
                "Use --dsv4: native mean-only final HS is not a valid teacher."
            )

    if args.target_layer_ids:
        target_layer_ids = args.target_layer_ids
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        if dsv4_manifest is not None:
            warnings.warn(
                f"DSV4 export slots: {target_layer_ids}. Pass only auxiliary IDs "
                f"{dsv4_manifest['auxiliary_hs_ids']} to training; "
                "slot 43 is the separate teacher.",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"Using custom target layer ids {target_layer_ids}. These "
                "must also be explicitly passed into the training script.",
                stacklevel=2,
            )
    else:
        target_layer_ids = [
            2,
            num_hidden_layers // 2,
            num_hidden_layers - 3,
            num_hidden_layers,
        ]

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    backend_cls = _backend_registry[args.hidden_states_backend]
    kv_transfer_config = backend_cls.build_kv_transfer_config(args)
    if args.dsv4_block_verify:
        from speculators_dsv4.block_protocol import (  # noqa: PLC0415
            BLOCK_CONNECTOR,
            BLOCK_CONNECTOR_MODULE,
        )

        kv_transfer_config = {
            "kv_connector": BLOCK_CONNECTOR,
            "kv_connector_module_path": BLOCK_CONNECTOR_MODULE,
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative_config",
        json.dumps(speculative_config),
        "--kv_transfer_config",
        json.dumps(kv_transfer_config),
        *vllm_args,
    ]

    disable_cp_arg = "--no-enable-chunked-prefill"
    if disable_cp_arg not in cmd:
        cmd.append(disable_cp_arg)

    print("Running command:")
    print(" ".join(cmd))

    if not args.dry_run:
        if dsv4_manifest is not None:
            ensure_manifest(
                args.hidden_states_path,
                dsv4_manifest,
                create=True,
                runtime_quantization=dsv4_runtime_quantization,
            )
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
