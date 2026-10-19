"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import os
import shutil
import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file

from hs_connectors.fp8_utils import SCALES_KEY, dequantize_fp8_tensor
from hs_connectors.mooncake_store import MooncakeHiddenStatesStore, MooncakeStoreConfig

if TYPE_CHECKING:
    from collections.abc import Callable


def wait_for_lock(lock_path: str, timeout: float = 10.0, poll_interval: float = 0.1):
    fd = os.open(lock_path, os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    strict_errors = False
    preserve_failed_artifacts = False

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    def request_kwargs(self) -> dict[str, Any]:
        """Optional request identity/retry policy; file consumers are unchanged."""
        return {}

    def close(self) -> None:  # noqa: B027
        """Release a preflight client before constructing dataloader workers."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

    return None


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return _load_hs_file(path)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return _load_hs_file(Path(handle))

    def cache(self, handle: str, file_idx: int) -> None:
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        target = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        shutil.move(handle, target)

    def delete(self, handle: str) -> None:
        Path(handle).unlink()


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        _add_argument_if_absent(
            parser,
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        _add_argument_if_absent(
            parser,
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }


# ---------------------------------------------------------------------------
# FP8 backend (shared filesystem, quantized hidden states)
# ---------------------------------------------------------------------------


class FP8Transfer(FileTransfer):
    """File-system transfer that transparently dequantizes FP8 payloads.

    Pairs with ``FP8HiddenStatesConnector`` on the vLLM side. Consumers see
    plain ``hidden_states`` tensors in ``dequantize_dtype`` -- the
    ``hidden_states_scales`` tensor and the quantization step are invisible
    to callers, mirroring ``FileTransfer``'s return shape exactly.
    """

    def __init__(
        self,
        hidden_states_path: Path,
        dequantize_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(hidden_states_path)
        self.dequantize_dtype = dequantize_dtype

    def _dequantize(
        self, sample: dict[str, torch.Tensor] | None
    ) -> dict[str, torch.Tensor] | None:
        if sample is None or SCALES_KEY not in sample:
            return sample
        scales = sample[SCALES_KEY]
        dequantized = dict(sample)
        dequantized["hidden_states"] = dequantize_fp8_tensor(
            sample["hidden_states"], scales, dtype=self.dequantize_dtype
        )
        del dequantized[SCALES_KEY]
        return dequantized

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        return self._dequantize(super().get_cached(file_idx))

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return self._dequantize(super().get_generated(handle))


def _add_argument_if_absent(
    parser: argparse.ArgumentParser, *args: Any, **kwargs: Any
) -> None:
    """Add an argument only if its dest is not already registered."""
    dest = kwargs.get("dest") or args[0].lstrip("-").replace("-", "_")
    if not any(a.dest == dest for a in parser._actions):  # noqa: SLF001
        parser.add_argument(*args, **kwargs)


@HiddenStatesBackend.register("fp8")
class FP8Backend(HiddenStatesBackend):
    """Shared-filesystem backend that quantizes hidden states to FP8.

    Reuses the same ``--hidden-states-path`` flag as ``FileBackend`` since
    the semantics are identical (a directory for safetensors files).
    """

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        _add_argument_if_absent(
            parser,
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        _add_argument_if_absent(
            parser,
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FP8Transfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FP8Transfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "FP8HiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": "hs_connectors.fp8_hidden_states_connector",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }


@HiddenStatesBackend.register("http")
class HttpBackend(HiddenStatesBackend):
    """Qwen3/DSV4 target files served by an authenticated training sidecar."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        FileBackend.add_train_args(parser)
        _add_argument_if_absent(
            parser,
            "--hs-http-endpoints",
            nargs="+",
            default=None,
            help="Authenticated HS sidecar URL(s), one per target host (not /v1).",
        )
        _add_argument_if_absent(
            parser,
            "--hs-http-timeout",
            type=float,
            default=120.0,
            help="HTTP HS readiness/download timeout, in seconds.",
        )
        _add_argument_if_absent(
            parser,
            "--hs-http-max-file-bytes",
            type=int,
            default=512 * 1024**2,
            help="Maximum bytes per downloaded HS artifact.",
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        FileBackend.add_launch_args(parser)

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,  # noqa: ARG004
    ) -> HiddenStatesTransfer:
        from speculators_dsv4 import HS_FORMAT  # noqa: PLC0415
        from speculators_dsv4.hs_training_http import (  # noqa: PLC0415
            HttpTrainingTransfer,
        )

        if getattr(args, "target_hidden_state_format", None) != HS_FORMAT:
            from speculators.remote_http import (  # noqa: PLC0415
                QwenHttpTrainingTransfer,
                checkpoint_contract,
            )

            if not args.hs_http_endpoints or not args.hidden_states_path:
                raise ValueError(
                    "HTTP HS requires --hs-http-endpoints and a trainer-local "
                    "--hidden-states-path download directory."
                )
            if getattr(args, "on_generate", "delete") != "delete":
                raise ValueError("Qwen HTTP HS requires --on-generate delete")
            return QwenHttpTrainingTransfer(
                args.hs_http_endpoints,
                os.environ.get("SPECULATORS_HS_HTTP_TOKEN"),
                args.hidden_states_path,
                timeout=args.hs_http_timeout,
                max_file_bytes=args.hs_http_max_file_bytes,
                expected_contract=checkpoint_contract(
                    args.verifier_name_or_path, args.target_layer_ids
                ),
            )
        if not args.hs_http_endpoints or not args.hidden_states_path:
            raise ValueError(
                "HTTP HS requires --hs-http-endpoints and a trainer-local "
                "--hidden-states-path download directory."
            )
        return HttpTrainingTransfer(
            args.hs_http_endpoints,
            os.environ.get("DSV4_HS_HTTP_TOKEN"),
            args.hidden_states_path,
            timeout=args.hs_http_timeout,
            max_file_bytes=args.hs_http_max_file_bytes,
            expected_contract=getattr(args, "target_training_contract", None),
        )

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        if not getattr(args, "dsv4", False):
            raise ValueError("The HTTP training backend currently requires --dsv4.")
        # Here the directory is target-local, never mounted by the trainer.
        args.hidden_states_path = str(Path(args.hidden_states_path).resolve())
        return FileBackend.build_kv_transfer_config(args)


# ---------------------------------------------------------------------------
# Mooncake-based backend (distributed store)
# ---------------------------------------------------------------------------


class MooncakeTransfer(HiddenStatesTransfer):
    """Mooncake distributed store based hidden-states transfer."""

    def __init__(self, store: MooncakeHiddenStatesStore):
        self.store = store

    def setup(self) -> None:
        if not self.store.is_setup:
            self.store.setup()

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:  # noqa: ARG002
        return None

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return self.store.get_sample(handle)

    def delete(self, handle: str) -> None:
        self.store.delete_sample(handle)


@HiddenStatesBackend.register("mooncake")
class MooncakeBackend(HiddenStatesBackend):
    """Mooncake distributed store backend (no shared filesystem required)."""

    @staticmethod
    def _add_mooncake_args(parser: argparse.ArgumentParser) -> None:
        _add_argument_if_absent(
            parser,
            "--mooncake-device-name",
            default="",
            help="Mooncake RDMA verbs device name(s); not the GLOO socket NIC.",
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-namespace",
            default=None,
            help="Fresh, shared run ID for DSV4 remote contracts and HS handles.",
        )
        _add_argument_if_absent(
            parser,
            "--hs-transfer-timeout",
            type=float,
            default=120.0,
            help="Mooncake contract/sample readiness timeout, in seconds.",
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-master",
            type=str,
            default="127.0.0.1:50051",
            help="Mooncake master server address. Used with backend=mooncake.",
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-metadata-server",
            type=str,
            default="P2PHANDSHAKE",
            help=(
                "Mooncake metadata server (or P2PHANDSHAKE). "
                "Used with backend=mooncake."
            ),
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-protocol",
            choices=["tcp", "rdma"],
            default="tcp",
            help="Mooncake transport protocol. Used with backend=mooncake.",
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-global-segment-gib",
            type=float,
            default=4.0,
            help=(
                "Memory registered by each Mooncake client for globally visible "
                "objects, in GiB. Increase for many concurrent long sequences."
            ),
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-local-buffer-gib",
            type=float,
            default=2.0,
            help="Mooncake client's local staging buffer, in GiB.",
        )

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)
        _add_argument_if_absent(
            parser,
            "--mooncake-target-dp-size",
            type=int,
            default=1,
            help="Global DSV4 target DP size; preflight checks every producer.",
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)
        _add_argument_if_absent(
            parser,
            "--mooncake-writer-threads",
            type=int,
            default=4,
            help="Number of asynchronous Mooncake writer threads in the vLLM client.",
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-pinned-pool-mib",
            type=int,
            default=0,
            help="Producer host-buffer reuse budget in MiB; 0 disables reuse.",
        )
        _add_argument_if_absent(
            parser,
            "--mooncake-copy-events",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Wait on each producer copy event instead of its stream.",
        )

    @staticmethod
    def _store_config(args, *, initialize_accelerator, **kwargs):
        return MooncakeStoreConfig(
            local_hostname=os.environ.get("MOONCAKE_LOCAL_HOSTNAME")
            or socket.gethostbyname(socket.gethostname()),
            metadata_server=args.mooncake_metadata_server,
            master_server_address=args.mooncake_master,
            global_segment_size=round(args.mooncake_global_segment_gib * 1024**3),
            local_buffer_size=round(args.mooncake_local_buffer_gib * 1024**3),
            protocol=args.mooncake_protocol,
            device_name=getattr(args, "mooncake_device_name", ""),
            initialize_accelerator=initialize_accelerator,
            **kwargs,
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,  # noqa: ARG004
    ) -> MooncakeTransfer:
        dsv4 = (
            getattr(args, "target_hidden_state_format", None)
            == "deepseek_v4_mean_hc_head"
        )
        store = MooncakeHiddenStatesStore(
            MooncakeBackend._store_config(args, initialize_accelerator=not dsv4)
        )
        if dsv4:
            from speculators_dsv4.mooncake_training import (  # noqa: PLC0415
                Dsv4MooncakeTransfer,
            )

            return Dsv4MooncakeTransfer(
                store,
                namespace=args.mooncake_namespace,
                target_dp_size=args.mooncake_target_dp_size,
                expected_contract=getattr(args, "target_training_contract", None),
                timeout=args.hs_transfer_timeout,
            )
        return MooncakeTransfer(store)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        mooncake_cfg = MooncakeBackend._store_config(
            args,
            initialize_accelerator=not getattr(args, "dsv4", False),
            num_writer_threads=args.mooncake_writer_threads,
            pinned_pool_bytes=getattr(args, "mooncake_pinned_pool_mib", 0) * 1024**2,
            use_copy_events=getattr(args, "mooncake_copy_events", False),
        )

        return {
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": (
                "hs_connectors.mooncake_hidden_states_connector"
            ),
            "kv_connector_extra_config": {
                "mooncake": dataclasses.asdict(mooncake_cfg),
            },
        }
