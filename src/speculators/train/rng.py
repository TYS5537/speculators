"""JSON-safe, per-rank snapshots of the training process's random generators.

Worker processes, prefetched data, independent Generator objects and remote data
services are outside this snapshot. Restoring it is not a bitwise-training claim.
"""

import base64
import logging
import random

import numpy as np
import torch
import torch.distributed as dist

# This captures the existing process-global NumPy RNG seeded by scripts/train.py,
# not a newly created independent Generator.
# ruff: noqa: NPY002

logger = logging.getLogger("speculators")
_VERSION = 1


def _gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _encode_tensor(state: torch.Tensor) -> str:
    return base64.b64encode(bytes(state.cpu().tolist())).decode("ascii")


def _decode_tensor(state: str) -> torch.Tensor:
    return torch.tensor(
        list(base64.b64decode(state, validate=True)), dtype=torch.uint8, device="cpu"
    )


def _capture_local(device_type: str) -> dict:
    py_version, py_state, py_gauss = random.getstate()
    algorithm, keys, position, has_gauss, cached_gauss = np.random.get_state()
    accelerator = None
    if device_type != "cpu":
        backend = getattr(torch, device_type)
        accelerator = _encode_tensor(backend.get_rng_state())
    return {
        "python": [py_version, list(py_state), py_gauss],
        "numpy": [algorithm, keys.tolist(), position, has_gauss, cached_gauss],
        "torch_cpu": _encode_tensor(torch.get_rng_state()),
        "accelerator": accelerator,
    }


def capture_rng_states(device_type: str) -> dict:
    """Collect each logical rank's state; all ranks must call this together.

    Only the current accelerator is read, not every visible CUDA/NPU device.
    Errors are exchanged before raising so one rank cannot leave peers waiting
    in the next checkpoint collective after a local capture failure.
    """
    try:
        result = {"state": _capture_local(device_type), "error": None}
    except Exception as exc:  # noqa: BLE001
        result = {"state": None, "error": f"{type(exc).__name__}: {exc}"}
    results = _gather(result)
    errors = [
        f"rank {i}: {item['error']}" for i, item in enumerate(results) if item["error"]
    ]
    if errors:
        raise RuntimeError("Cannot capture checkpoint RNG states: " + "; ".join(errors))
    return {
        "version": _VERSION,
        "world_size": len(results),
        "device_type": device_type,
        "states": [item["state"] for item in results],
    }


def _prepare_local(state: dict, device_type: str) -> tuple:
    """Validate CPU states using isolated generators before touching global ones."""
    py_version, py_state, py_gauss = state["python"]
    python_state = (py_version, tuple(py_state), py_gauss)
    random.Random().setstate(python_state)
    algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
    numpy_state = (
        algorithm,
        np.asarray(keys, dtype=np.uint32),
        position,
        has_gauss,
        cached_gauss,
    )
    np.random.RandomState(0).set_state(numpy_state)
    cpu_state = _decode_tensor(state["torch_cpu"])
    torch.Generator(device="cpu").set_state(cpu_state)
    accelerator = state["accelerator"]
    if (accelerator is None) != (device_type == "cpu"):
        raise ValueError("Checkpoint RNG accelerator state does not match device_type")
    accelerator_state = None if accelerator is None else _decode_tensor(accelerator)
    return python_state, numpy_state, cpu_state, accelerator_state


def _apply_local(prepared: tuple, device_type: str) -> None:
    python_state, numpy_state, cpu_state, accelerator_state = prepared
    # Set the accelerator first; lazy backend initialization must not advance the
    # CPU/Python generators after their checkpoint state has been restored.
    if accelerator_state is not None:
        getattr(torch, device_type).set_rng_state(accelerator_state)
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(cpu_state)


def _compatibility(
    payload: dict | None, device_type: str, world_size: int
) -> str | None:
    if payload is None:
        return (
            "Checkpoint has no RNG states (legacy format); keeping startup RNG states"
        )
    if payload["version"] != _VERSION:
        raise ValueError(f"Unsupported checkpoint RNG version: {payload['version']}")
    if payload["world_size"] != world_size or payload["device_type"] != device_type:
        return (
            "Checkpoint RNG topology/backend differs from the current run; "
            "keeping startup RNG states, without exact random-stream replay"
        )
    if len(payload["states"]) != world_size:
        raise ValueError("Checkpoint RNG rank-state count does not match world_size")
    return None


def restore_rng_states(payload: dict | None, device_type: str) -> bool:
    """Restore this rank after setup (and mid-epoch iterator/skip initialization).

    Missing legacy state or a changed topology/backend is a warned compatibility
    fallback. Malformed state fails on all ranks. All ranks must call together.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    prepared = backup = None
    try:
        reason = _compatibility(payload, device_type, world_size)
        if reason is None:
            prepared = _prepare_local(payload["states"][rank], device_type)
            backup = _prepare_local(_capture_local(device_type), device_type)
        result = {"reason": reason, "error": None}
    except Exception as exc:  # noqa: BLE001
        result = {"reason": None, "error": f"{type(exc).__name__}: {exc}"}
    results = _gather(result)
    errors = [
        f"rank {i}: {item['error']}" for i, item in enumerate(results) if item["error"]
    ]
    if errors:
        raise RuntimeError("Cannot restore checkpoint RNG states: " + "; ".join(errors))
    reasons = {item["reason"] for item in results if item["reason"]}
    if reasons:
        for reason in sorted(reasons):
            logger.warning(reason)
        return False

    try:
        _apply_local(prepared, device_type)
        error = None
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    errors = _gather(error)
    if any(errors):
        # Do not leave successful ranks on a new stream while a peer failed.
        try:
            _apply_local(backup, device_type)
            rollback_error = None
        except Exception as exc:  # noqa: BLE001
            rollback_error = f"{type(exc).__name__}: {exc}"
        rollback_errors = _gather(rollback_error)
        raise RuntimeError(
            f"Failed to apply checkpoint RNG states: {errors}; "
            f"rollback errors: {rollback_errors}"
        )
    return True
