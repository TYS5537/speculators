import filecmp
import functools
import json
import logging
import shutil
import time
import uuid
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.parallel import DistributedDataParallel
from transformers.modeling_utils import PreTrainedModel

from speculators.train.distributed import get_rank, is_distributed
from speculators.utils.util import get_current_device

logger = logging.getLogger("speculators")

# Optimizers/schedulers may be a single object (legacy) or a list (e.g. Muon + AdamW).
OptimizerOrList = torch.optim.Optimizer | list[torch.optim.Optimizer]
SchedulerOrList = (
    torch.optim.lr_scheduler.LRScheduler | list[torch.optim.lr_scheduler.LRScheduler]
)


def _as_list(value):
    """Normalize a single object or a list/tuple of objects into a list."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _model_weight_files(directory: Path) -> list[str]:
    if (directory / "model.safetensors").is_file():
        return ["model.safetensors"]
    index_name = "model.safetensors.index.json"
    index = json.loads((directory / index_name).read_text())
    weights = index["weight_map"]
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Invalid model shard index")
    shards = set(weights.values())
    if any(
        not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".safetensors")
        for name in shards
    ):
        raise ValueError("Invalid model shard path")
    return [index_name, *sorted(shards)]


def _rank0_only(fn):
    """Run rank-zero I/O and propagate its outcome before peers continue."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result, error = None, None
        if get_rank() == 0:
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                if not is_distributed():
                    raise
                error = f"{type(exc).__name__}: {exc}"
        if is_distributed():
            outcome = [result, error]
            dist.broadcast_object_list(outcome, src=0)
            result, error = outcome
        if error is not None:
            raise RuntimeError(f"Checkpoint {fn.__name__} failed on rank 0: {error}")
        return result

    return wrapper


class BaseCheckpointer:
    """Helper class to save and load checkpoints.

    Checkpoint file structure:
    ../path/
        0/ # epoch number
            model.safetensors
            optimizer_state_dict.pt
            scheduler_state_dict.pt (optional)
        1/
            model.safetensors
            optimizer_state_dict.pt
            scheduler_state_dict.pt (optional)
        ...
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.previous_epoch, self.prev_path = self._find_previous_checkpoint()

    @abstractmethod
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        raise NotImplementedError

    @abstractmethod
    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        raise NotImplementedError

    def load_scheduler_state_dict(self, scheduler: SchedulerOrList):
        scheduler_path = self.scheduler_path(self.previous_epoch)
        if not scheduler_path.exists():
            return
        loaded = torch.load(scheduler_path, weights_only=True)
        schedulers = _as_list(scheduler)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        for sched, state_dict in zip(schedulers, loaded_list, strict=True):
            sched.load_state_dict(state_dict)
            # Constructing a scheduler performs its initial step and can overwrite
            # the optimizer LR loaded just before it. load_state_dict restores
            # scheduler counters only; restore the recorded LR before the first
            # resumed optimizer step (including each Muon/AdamW optimizer).
            if "_last_lr" in state_dict:
                for group, lr in zip(
                    sched.optimizer.param_groups, sched.get_last_lr(), strict=True
                ):
                    if isinstance(group["lr"], torch.Tensor):
                        group["lr"].fill_(lr)
                    else:
                        group["lr"] = lr

    @_rank0_only
    def save_scheduler_state_dict(self, scheduler: SchedulerOrList, epoch: int | str):
        schedulers = _as_list(scheduler)
        state_dicts = [sched.state_dict() for sched in schedulers]
        # Preserve the legacy single-scheduler format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.scheduler_path(epoch))

    @abstractmethod
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raise NotImplementedError

    COMMIT_FILENAME = "checkpoint_complete.json"

    @staticmethod
    def _bundle_files(path: Path, has_scheduler: bool) -> list[str]:
        required = ["config.json", "optimizer_state_dict.pt", "training_state.json"]
        if has_scheduler:
            required.append("scheduler_state_dict.pt")
        required.extend(_model_weight_files(path))
        for name in required:
            item = path / name
            if not item.is_file() or item.stat().st_size == 0:
                raise FileNotFoundError(f"Incomplete checkpoint: {item}")
        return required

    @staticmethod
    def _validate_training_state(state: dict) -> None:
        if not isinstance(state, dict):
            raise ValueError("Invalid checkpoint progress")
        if any(
            type(state.get(key)) is not int or state[key] < 0
            for key in ("epoch", "local_step", "global_step")
        ):
            raise ValueError(
                "Checkpoint progress must contain nonnegative integer counters"
            )
        if not isinstance(state.get("epoch_complete"), bool):
            raise ValueError("Checkpoint progress must specify epoch_complete")
        if "epoch_finalized" in state and not isinstance(
            state["epoch_finalized"], bool
        ):
            raise ValueError("Checkpoint epoch_finalized must be boolean")

    def _read_committed_progress(self, path: Path, manifest: dict) -> dict:
        committed = manifest["training_state"]
        state = json.loads((path / "training_state.json").read_text())
        self._validate_training_state(committed)
        self._validate_training_state(state)
        # Finishing validation/best bookkeeping is an atomic metadata-only write.
        # It may advance this flag, but may never change the checkpoint's weights,
        # counters or training-complete boundary recorded by the manifest.
        expected = dict(committed)
        if (
            committed.get("epoch_finalized") is False
            and state.get("epoch_finalized") is True
        ):
            if not committed["epoch_complete"]:
                raise ValueError("Cannot finalize an unfinished training epoch")
            expected["epoch_finalized"] = True
            if "rng_states" in state:
                expected["rng_states"] = state["rng_states"]
        if state != expected:
            raise ValueError("Checkpoint progress does not match its manifest")
        return state

    def _checkpoint_candidate(self, path: Path):
        """Ignore unpublished stages; accept committed bundles and legacy epochs."""
        if path.is_symlink() or not path.is_dir():
            return None
        label = path.name
        if label.startswith(".previous-"):
            label = label.removeprefix(".previous-").rsplit("-", 1)[0]
        if label != "interrupted" and not label.isdecimal():
            return None
        if (path / self.COMMIT_FILENAME).exists():
            return self._committed_candidate(path)
        # Old interrupted snapshots have no reliable epoch/progress metadata.
        # Numeric legacy checkpoints remain readable without a new manifest.
        if label.isdecimal():
            return (0, int(label), int(label), path)
        logger.warning(
            "Ignoring legacy interrupted checkpoint without metadata: %s", path
        )
        return None

    def _committed_candidate(self, path: Path):
        try:
            manifest = json.loads((path / self.COMMIT_FILENAME).read_text())
            if manifest["version"] != 1 or not isinstance(
                manifest["has_scheduler"], bool
            ):
                return None
            state = self._read_committed_progress(path, manifest)
            self._bundle_files(path, manifest["has_scheduler"])
            return (1, int(manifest["created_ns"]), int(state["epoch"]), path)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Ignoring incomplete checkpoint %s", path)
            return None

    def _find_previous_checkpoint(self) -> tuple[int, Path | None]:
        if not self.path.exists():
            return -1, None
        candidates = [
            candidate
            for child in self.path.iterdir()
            if (candidate := self._checkpoint_candidate(child)) is not None
        ]
        if not candidates:
            return -1, None
        latest = max(
            candidates, key=lambda entry: (*entry[:2], entry[3].name.isdecimal())
        )
        return latest[2], latest[3]

    def _get_previous_epoch(self) -> int:
        return self._find_previous_checkpoint()[0]

    def _load_path(self, epoch: int | str) -> Path:
        if epoch == self.previous_epoch and self.prev_path is not None:
            return self.prev_path
        return self.path / str(epoch)

    @_rank0_only
    def _prepare_checkpoint_stage(self, label: str) -> str:
        if label != "interrupted" and not label.isdecimal():
            raise ValueError(f"Invalid checkpoint label: {label!r}")
        stage = self.path / f".pending-{uuid.uuid4().hex}"
        (stage / label).mkdir(parents=True)
        command = self.path / self.TRAIN_COMMAND_FILENAME
        if command.is_file():
            shutil.copy2(command, stage / self.TRAIN_COMMAND_FILENAME)
        return str(stage)

    @_rank0_only
    def _publish_checkpoint(self, stage: Path, label: str, has_scheduler: bool):
        source = stage / label
        required = self._bundle_files(source, has_scheduler)
        state = json.loads((source / "training_state.json").read_text())
        self._validate_training_state(state)
        manifest = {
            "version": 1,
            "created_ns": time.time_ns(),
            "has_scheduler": has_scheduler,
            "training_state": state,
            "files": required,
        }
        (source / self.COMMIT_FILENAME).write_text(json.dumps(manifest))
        destination = self.path / label
        backup = self.path / f".previous-{label}-{uuid.uuid4().hex}"
        best_snapshot = (
            self._best_checkpoint_path()
            if str(self.read_best_epoch()) == label
            else None
        )
        if destination.exists():
            destination.rename(backup)
            if best_snapshot == destination:
                best_snapshot = backup
        try:
            if best_snapshot is not None:
                self._set_best_target(best_snapshot)
            source.rename(destination)
        except BaseException:
            # If best already references the displaced generation, leave that
            # exact complete directory in place. Automatic resume finds it too.
            if (
                backup.exists()
                and not destination.exists()
                and self._best_link_target() != backup
            ):
                backup.rename(destination)
            raise
        self._prune_recovery_generations(label)
        shutil.rmtree(stage)

    def _prune_recovery_generations(self, label: str):
        # If startup fell back to a backup, the displaced current directory may
        # be corrupt or missing. Preserve the latest *valid* old generation,
        # rather than unconditionally retaining the just-displaced directory.
        backups = list(self.path.glob(f".previous-{label}-*"))
        candidates = [
            candidate
            for path in backups
            if (candidate := self._checkpoint_candidate(path)) is not None
        ]
        keep = max(candidates, key=lambda item: item[:2])[3] if candidates else None
        best_snapshot = self._best_link_target()
        for old in backups:
            if (
                old not in (keep, best_snapshot)
                and old.is_dir()
                and not old.is_symlink()
            ):
                shutil.rmtree(old)

    @_rank0_only
    def mark_epoch_finalized(
        self, epoch: int, global_step: int, *, rng_states: dict | None = None
    ):
        """Atomically mark successful validation/best bookkeeping on this snapshot."""
        saved_epoch, path = self._find_previous_checkpoint()
        if path is None or saved_epoch != epoch:
            return
        state_path = path / "training_state.json"
        if not state_path.is_file():
            return  # Legacy checkpoints may not have progress metadata.
        state = json.loads(state_path.read_text())
        if (
            state.get("global_step") != global_step
            or not state.get("epoch_complete")
            or state.get("epoch_finalized") is not False
        ):
            return
        state["epoch_finalized"] = True
        if rng_states is not None:
            state["rng_states"] = rng_states
        pending = path / f".training-state-{uuid.uuid4().hex}.json"
        pending.write_text(json.dumps(state))
        pending.replace(state_path)

    @_rank0_only
    def can_reuse_validation_checkpoint(self, epoch: int, global_step: int) -> bool:
        """Reuse only an intact numeric snapshot of the restored model's weights.

        A validation-only restart must not discard that snapshot's evaluation
        metadata. Matching step counters alone does not establish weight identity,
        particularly when recovering an interrupted or previous generation.
        """
        source, destination = self.prev_path, self.path / str(epoch)
        if source is None:
            return False
        for path in (source, destination):
            if self._checkpoint_candidate(path) is None:
                return False
            state_path = path / "training_state.json"
            if not state_path.is_file():
                return False
            state = json.loads(state_path.read_text())
            if (
                state.get("epoch") != epoch
                or state.get("global_step") != global_step
                or not state.get("epoch_complete")
            ):
                return False
        if source == destination:
            return True
        source_files = _model_weight_files(source)
        return source_files == _model_weight_files(destination) and all(
            filecmp.cmp(source / name, destination / name, shallow=False)
            for name in ["config.json", *source_files]
        )

    @contextmanager
    def checkpoint_transaction(self, epoch: int | str, *, has_scheduler: bool):
        """Publish model, optimizer, scheduler and progress as one complete bundle.

        Directory replacement keeps the old generation discoverable until the
        new directory is published, including interruption between the renames.
        Pending directories are never candidates for automatic restoration.
        """
        label = str(epoch)
        root, previous = self.path, self.prev_path
        stage = Path(self._prepare_checkpoint_stage(label))
        self.path, self.prev_path = stage, None
        try:
            yield
        finally:
            self.path, self.prev_path = root, previous
        self._publish_checkpoint(stage, label, has_scheduler)

    def model_path(self, epoch: int | str):
        model_fname = "model.safetensors"
        return self._load_path(epoch) / model_fname

    def optimizer_path(self, epoch: int | str):
        optimizer_fname = "optimizer_state_dict.pt"
        return self._load_path(epoch) / optimizer_fname

    def scheduler_path(self, epoch: int | str):
        scheduler_fname = "scheduler_state_dict.pt"
        return self._load_path(epoch) / scheduler_fname

    def best_path(self) -> Path:
        return self.path / "checkpoint_best"

    def val_metrics_path(self, epoch: int) -> Path:
        return self.path / str(epoch) / "val_metrics.json"

    TRAIN_COMMAND_FILENAME = "train_command.txt"

    def _copy_train_command(self, epoch: int | str) -> None:
        src = self.path / self.TRAIN_COMMAND_FILENAME
        if src.exists():
            shutil.copy2(src, self.path / str(epoch) / self.TRAIN_COMMAND_FILENAME)

    @_rank0_only
    def save_val_metrics(self, epoch: int, val_metrics: dict[str, float]):
        path = self.val_metrics_path(epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(val_metrics))

    def load_best_val_loss(self) -> float | None:
        best_snapshot = self._best_checkpoint_path()
        if best_snapshot is None:
            return None
        p = best_snapshot / "val_metrics.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            return float(data["loss_epoch"])
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    def read_best_epoch(self) -> int | None:
        """Return the epoch that `checkpoint_best` points to."""
        target = self._best_link_target()
        if target is None:
            return None
        label = target.name
        if label.startswith(".previous-"):
            label = label.removeprefix(".previous-").rsplit("-", 1)[0]
        try:
            return int(label)
        except ValueError:
            return None

    def _best_link_target(self) -> Path | None:
        best = self.best_path()
        if not best.is_symlink():
            return None
        try:
            return self.path / best.readlink().name
        except OSError:
            return None

    def _best_checkpoint_path(self) -> Path | None:
        target = self._best_link_target()
        if target is not None and self._checkpoint_candidate(target) is not None:
            return target
        epoch = self.read_best_epoch()
        # A numeric target may temporarily be displaced, or recovered from its
        # previous generation. An explicit generation target must never silently
        # switch to a different model just because its epoch number matches.
        if epoch is None or (
            target is not None and target.name.startswith(".previous-")
        ):
            return None
        candidate = self._epoch_checkpoint_path(epoch)
        return candidate if self._checkpoint_candidate(candidate) is not None else None

    def _epoch_checkpoint_path(self, epoch: int) -> Path:
        candidates = [
            candidate
            for child in self.path.iterdir()
            if (
                child.name == str(epoch) or child.name.startswith(f".previous-{epoch}-")
            )
            and (candidate := self._checkpoint_candidate(child)) is not None
        ]
        if not candidates:
            return self.path / str(epoch)
        return max(
            candidates, key=lambda entry: (*entry[:2], entry[3].name.isdecimal())
        )[3]

    def load_model_state_dict_for_epoch(
        self, model: PreTrainedModel, epoch: int, float_dtype: torch.dtype | None = None
    ):
        """Temporarily load weights for a specific epoch."""
        old_epoch, old_path = self.previous_epoch, self.prev_path
        try:
            self.previous_epoch = epoch
            self.prev_path = self._epoch_checkpoint_path(epoch)
            self.load_model_state_dict(model, float_dtype=float_dtype)
        finally:
            self.previous_epoch = old_epoch
            self.prev_path = old_path

    @_rank0_only
    def update_best_symlink(self, epoch: int):
        self._set_best_target(self.path / str(epoch))

    def _set_best_target(self, target: Path):
        """Atomically move the best pointer without changing its chosen weights."""
        best_path = self.best_path()
        pending = self.path / f".checkpoint-best-{uuid.uuid4().hex}"
        try:
            pending.symlink_to(Path(target.name), target_is_directory=True)
            if best_path.is_dir() and not best_path.is_symlink():
                shutil.rmtree(best_path)
            pending.replace(best_path)
        finally:
            if pending.is_symlink():
                pending.unlink()

    @_rank0_only
    def cleanup_keep_only_best(self, best_epoch: int) -> None:
        """
        Delete all epoch dir. except best_epoch, and keep best_checkpoint symlink.
        """
        keep_dir = self.path / str(best_epoch)
        best_link = self.best_path()

        # Safety checks
        if not keep_dir.exists() or not keep_dir.is_dir():
            raise FileNotFoundError(f"Best epoch dir does not exist: {keep_dir}")

        train_cmd_file = self.path / self.TRAIN_COMMAND_FILENAME

        for child in self.path.iterdir():
            # Keep the symlink itself
            if child == best_link:
                continue

            # Keep the best epoch directory
            if child == keep_dir:
                continue

            if child == train_cmd_file:
                continue

            # Transaction recovery generations must survive best-only cleanup.
            if child.name.startswith((".previous-", ".pending-")):
                continue

            # Delete numbered epoch directories and any other stray dirs/files
            try:
                if child.is_symlink() or child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
            except (FileNotFoundError, PermissionError, OSError) as exc:
                raise RuntimeError(f"Failed to delete {child}") from exc


def convert_float_dtype(sd: pytree.PyTree, dtype: torch.dtype) -> pytree.PyTree:
    def convert_fn(x):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x.to(dtype)
        return x

    return pytree.tree_map(convert_fn, sd)


def convert_optimizer_state_dtype(
    sd: pytree.PyTree, dtype: torch.dtype
) -> pytree.PyTree:
    """Convert moments without quantizing optimizer step counters.

    Supports nested/list optimizer states and DCP's flattened ``state.*.step``
    keys. Older low-precision counters are promoted so subsequent increments
    work, but their already-rounded values cannot be reconstructed.
    """

    def convert_fn(path, value):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            return value
        key = path[-1].key if path and isinstance(path[-1], pytree.MappingKey) else None
        if isinstance(key, str) and (key == "step" or key.endswith(".step")):
            if value.dtype in (torch.float16, torch.bfloat16):
                return value.float()
            return value
        return value.to(dtype)

    return pytree.tree_map_with_path(convert_fn, sd)


def load_safetensors_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    full_state_dict = {}
    weight_files = _model_weight_files(path.parent)
    for name in weight_files:
        if not name.endswith(".safetensors"):
            continue
        with safe_open(path.parent / name, framework="pt", device=device) as f:
            for key in f.keys():  # noqa: SIM118
                full_state_dict[key] = f.get_tensor(key)
    return full_state_dict


def patch_config_dtype(config_path: Path, float_dtype: torch.dtype) -> None:
    """Patch config.json to match the actual on-disk tensor dtype.

    When models are kept in FP32 but saved as BF16, save_pretrained writes
    the in-memory dtype to config.json. This patches it to match the saved dtype.
    """
    if not config_path.exists():
        return

    config = json.loads(config_path.read_text())
    # Convert torch.bfloat16 -> "bfloat16"
    dtype_str = str(float_dtype).split(".")[-1]
    # Support both dtype (transformers 5.x) and torch_dtype (older versions)
    if "dtype" in config:
        config["dtype"] = dtype_str
    if "torch_dtype" in config:
        config["torch_dtype"] = dtype_str
    config_path.write_text(json.dumps(config, indent=2) + "\n")


class SingleGPUCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        device = get_current_device()
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch),
            device,
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )
        # Note: `strict=False` because we don't load the verifier weights
        model.load_state_dict(full_state_dict, strict=False)

    def load_optimizer_state_dict(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        device = get_current_device()
        loaded = torch.load(
            self.optimizer_path(self.previous_epoch),
            weights_only=True,
            map_location=device,
        )
        optimizers = _as_list(optimizer)
        loaded_list = loaded if isinstance(loaded, list) else [loaded]
        raw_model = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        dtype = float_dtype or raw_model.dtype
        for opt, state_dict in zip(optimizers, loaded_list, strict=True):
            opt.load_state_dict(convert_optimizer_state_dtype(state_dict, dtype))

    @_rank0_only
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        raw_model: PreTrainedModel = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )  # type: ignore[assignment]
        model_state_dict = convert_float_dtype(raw_model.state_dict(), float_dtype)
        raw_model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
        patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)

        optimizers = _as_list(optimizer)
        state_dicts = [
            convert_optimizer_state_dtype(opt.state_dict(), float_dtype)
            for opt in optimizers
        ]
        # Preserve the legacy single-optimizer format when there is only one.
        payload = state_dicts[0] if len(state_dicts) == 1 else state_dicts
        torch.save(payload, self.optimizer_path(epoch))
        self._copy_train_command(epoch)


class DistributedCheckpointer(BaseCheckpointer):
    def load_model_state_dict(
        self, model: PreTrainedModel, float_dtype: torch.dtype | None = None
    ):
        full_state_dict = load_safetensors_state_dict(
            self.model_path(self.previous_epoch), "cpu"
        )
        full_state_dict = convert_float_dtype(
            full_state_dict, float_dtype or model.dtype
        )

        # Note: `strict=False` because we don't load the verifier weights
        set_model_state_dict(
            model,
            full_state_dict,  # type: ignore[arg-type]
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )
        dist.barrier()

    def load_optimizer_state_dict(
        self,
        model,
        optimizer: OptimizerOrList,
        float_dtype: torch.dtype | None = None,
    ):
        optimizers = _as_list(optimizer)
        full_state_dict = torch.load(
            self.optimizer_path(self.previous_epoch),
            mmap=True,
            weights_only=True,
            map_location="cpu",
        )
        full_state_dict = convert_optimizer_state_dtype(
            full_state_dict, float_dtype or model.dtype
        )

        set_optimizer_state_dict(
            model,
            optimizers,
            full_state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

        # Keep counters at least FP32 if the state loader returned a low-precision
        # scalar. Do not downcast counters that already have higher precision.
        for opt in optimizers:
            for state in opt.state.values():
                step = state.get("step")
                if isinstance(step, torch.Tensor) and step.dtype in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    state["step"] = step.float()

        dist.barrier()

    def save_checkpoint(
        self,
        model: PreTrainedModel,
        optimizer: OptimizerOrList,
        epoch: int | str,
        float_dtype: torch.dtype = torch.bfloat16,
    ):
        model_state_dict = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        model_state_dict = convert_float_dtype(model_state_dict, float_dtype)

        optimizer_state_dict = get_optimizer_state_dict(
            model,
            _as_list(optimizer),
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        optimizer_state_dict = convert_optimizer_state_dtype(
            optimizer_state_dict, float_dtype
        )

        self._save_full_state(
            model, model_state_dict, optimizer_state_dict, epoch, float_dtype
        )

    @_rank0_only
    def _save_full_state(
        self, model, model_state_dict, optimizer_state_dict, epoch, float_dtype
    ):
        model.save_pretrained(self.path / str(epoch), state_dict=model_state_dict)
        patch_config_dtype(self.path / str(epoch) / "config.json", float_dtype)
        torch.save(optimizer_state_dict, self.optimizer_path(epoch))
        self._copy_train_command(epoch)
