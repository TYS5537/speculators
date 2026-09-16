"""Bounded acceptance checks around the real training/save/resume implementation.

The factory keeps heavyweight training imports out of the command's help path.
It does not emulate an NPU run: the command requires an NPU, while unit tests
exercise these checks with tiny CPU models. Checkpoint probes are sampled, not
a claim of bitwise equality for every tensor or uninterrupted RNG trajectories.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmokeSettings:
    phase: str
    report_dir: Path
    train_batches: int = 2
    val_batches: int = 1
    recipe_sha256: str = ""

    def __post_init__(self):
        if self.phase not in {"fresh", "resume"}:
            raise ValueError("smoke phase must be fresh or resume")
        if self.train_batches < 1 or self.val_batches < 1:
            raise ValueError("smoke train/validation batch counts must be positive")


def recipe_digest(arguments: list[str]) -> str:
    return hashlib.sha256(json.dumps(arguments).encode("utf-8")).hexdigest()


class BoundedLoader:
    """Cap batches without replacing the production sampler or collator."""

    def __init__(self, loader, limit: int):
        if loader is None or limit < 1 or len(loader) < 1:
            raise ValueError("smoke needs nonempty train AND validation loaders")
        self.loader = loader
        self.limit = min(limit, len(loader))

    def __len__(self):
        return self.limit

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.limit)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def _probe(value, torch):
    if isinstance(value, torch.Tensor):
        sample = value.detach().reshape(-1)[:8]
        if sample.is_floating_point():
            # Production checkpoints intentionally serialize floats as BF16.
            sample = sample.to(torch.bfloat16).float()
        return {"shape": list(value.shape), "sample": sample.cpu().tolist()}
    if isinstance(value, dict):
        return {str(key): _probe(val, torch) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_probe(val, torch) for val in value]
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError(f"unsupported smoke checkpoint probe: {type(value).__name__}")


def state_probe(trainer, torch) -> dict[str, Any]:
    """Small named samples of all model/state tensors plus full scheduler metadata."""
    named = dict(trainer.model.named_parameters())
    names = {id(param): name for name, param in named.items()}
    optimizers = []
    for optimizer in trainer.optimizers:
        optimizers.append(
            {
                "class": type(optimizer).__name__,
                "groups": [
                    {
                        **{
                            key: _probe(value, torch)
                            for key, value in group.items()
                            if key != "params"
                        },
                        "params": [names[id(param)] for param in group["params"]],
                    }
                    for group in optimizer.param_groups
                ],
                "state": {
                    names[id(param)]: _probe(state, torch)
                    for param, state in optimizer.state.items()
                },
            }
        )
    return {
        "model": {name: _probe(param, torch) for name, param in named.items()},
        "optimizers": optimizers,
        "schedulers": [_probe(s.state_dict(), torch) for s in trainer.schedulers],
        "global_step": trainer.global_step,
    }


def verify_restored_state(previous: dict, current: dict) -> None:
    for key in ("model", "optimizers", "schedulers", "global_step"):
        if previous[key] != current[key]:
            raise ValueError(
                f"smoke resume did not restore {key} (sampled tensor probes)"
            )


def _validate_phase_paths(settings: SmokeSettings, checkpoint_dir: Path, rank: int):
    report = settings.report_dir / f"{settings.phase}.rank-{rank}.json"
    if report.exists():
        raise ValueError(
            f"smoke report already exists; use a new run directory: {report}"
        )
    numbered = (
        sorted(
            child.name
            for child in checkpoint_dir.iterdir()
            if child.is_dir() and not child.is_symlink() and child.name.isdecimal()
        )
        if checkpoint_dir.exists()
        else []
    )
    if settings.phase == "fresh":
        if numbered:
            raise ValueError("fresh smoke refuses an existing training checkpoint")
        return None
    fresh_path = settings.report_dir / f"fresh.rank-{rank}.json"
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    if fresh.get("status") != "passed":
        raise ValueError("resume smoke requires a successful fresh-phase report")
    if fresh.get("recipe_sha256") != settings.recipe_sha256:
        raise ValueError("smoke recipe arguments changed between fresh and resume")
    if fresh.get("checkpoint_dir") != str(checkpoint_dir.resolve()):
        raise ValueError("smoke resume checkpoint directory changed")
    if numbered != ["0"]:
        raise ValueError("resume smoke expects exactly the fresh epoch-0 checkpoint")
    for name in (
        "config.json",
        "model.safetensors",
        "optimizer_state_dict.pt",
        "scheduler_state_dict.pt",
        "training_state.json",
    ):
        if not (checkpoint_dir / "0" / name).is_file():
            raise ValueError(f"smoke checkpoint is missing {name}")
    return fresh


def make_smoke_trainer(base_class, settings: SmokeSettings):  # noqa: C901
    """Use real model, losses, optimizers, validation and checkpoint paths."""
    import torch  # noqa: PLC0415
    import torch.distributed as dist  # noqa: PLC0415

    class SmokeTrainer(base_class):
        def _collective_device(self):
            # Model parameters are still on CPU during preflight. HCCL/NCCL
            # flags must nevertheless use the accelerator selected by torchrun.
            backend = str(dist.get_backend()) if dist.is_initialized() else ""
            if backend == "hccl":
                return torch.device("npu", torch.npu.current_device())
            if backend == "nccl":
                return torch.device("cuda", torch.cuda.current_device())
            return next(self.model.parameters()).device

        def _check(self, error: str | None) -> None:
            # Every rank participates before raising, including rank-local
            # validation/probe failures, so peers do not enter a later collective.
            if dist.is_available() and dist.is_initialized():
                device = self._collective_device()
                flag = torch.tensor(int(error is not None), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item() and error is None:
                    error = "smoke check failed on another training rank"
            if error:
                raise ValueError(error)

        def __init__(self, model, config, train_loader, val_loader=None):  # noqa: C901
            self.model = model
            rank = dist.get_rank() if dist.is_initialized() else 0
            self._smoke_report: dict[str, Any] = {
                "phase": settings.phase,
                "rank": rank,
                "device": str(next(model.parameters()).device),
                "recipe_sha256": settings.recipe_sha256,
                "checkpoint_dir": str(Path(config.save_path).resolve()),
                "probe_scope": "first 8 values of each tensor, canonicalized to BF16",
            }
            error = None
            fresh = None
            try:
                if config.fsdp_shard:
                    raise ValueError(
                        "this short acceptance harness supports DDP, not FSDP"
                    )
                if config.optimizer != "muon" or config.scheduler_type != "linear":
                    raise ValueError("DSV4 recipe smoke expects Muon + linear")
                train_loader = BoundedLoader(train_loader, settings.train_batches)
                val_loader = BoundedLoader(val_loader, settings.val_batches)
                if any(
                    getattr(loader, "num_workers", 0)
                    for loader in (train_loader, val_loader)
                ):
                    raise ValueError("smoke loaders must disable worker prefetch")
                fresh = _validate_phase_paths(settings, Path(config.save_path), rank)
                if fresh and (
                    fresh["train_batches"] != len(train_loader)
                    or fresh["val_batches"] != len(val_loader)
                ):
                    raise ValueError("smoke data loader lengths changed on resume")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            self._check(error)
            if dist.is_initialized():
                sizes = torch.tensor(
                    [len(train_loader), len(val_loader)],
                    device=self._collective_device(),
                )
                smallest, largest = sizes.clone(), sizes.clone()
                dist.all_reduce(smallest, op=dist.ReduceOp.MIN)
                dist.all_reduce(largest, op=dist.ReduceOp.MAX)
                self._check(
                    None
                    if torch.equal(smallest, largest)
                    else "smoke loader batch counts differ across ranks"
                )
            config = config._replace(
                num_epochs=1 if settings.phase == "fresh" else 2,
                resume_from_checkpoint=settings.phase == "resume",
                scheduler_total_steps=2 * len(train_loader),
                scheduler_warmup_steps=0,
                scheduler_warmup_ratio=None,
                checkpoint_freq=1,
                save_best=False,
                log_freq=1,
            )
            super().__init__(model, config, train_loader, val_loader)
            self._smoke_report.update(
                device=str(next(self.model.parameters()).device),
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
                train_batches=len(train_loader),
                val_batches=len(val_loader),
                overrides={
                    "epochs": config.num_epochs,
                    "resume_from_checkpoint": config.resume_from_checkpoint,
                    "scheduler_total_steps": config.scheduler_total_steps,
                    "scheduler_warmup_steps": 0,
                    "checkpoint_freq": 1,
                    "save_best": False,
                    "log_freq": 1,
                    "num_workers": 0,
                },
            )
            initial = state_probe(self, torch)
            error = None
            try:
                expected_epoch = 0 if settings.phase == "fresh" else 1
                if self.current_epoch != expected_epoch:
                    raise ValueError("smoke resumed at an unexpected epoch")
                if fresh:
                    if fresh["world_size"] != self._smoke_report["world_size"]:
                        raise ValueError("smoke world size changed on resume")
                    verify_restored_state(fresh["final"], initial)
                elif self.global_step != 0:
                    raise ValueError("fresh smoke must start at global_step=0")
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
            self._check(error)
            self._smoke_report["initial"] = initial
            self._smoke_train_calls = self._smoke_val_calls = self._smoke_grad_steps = 0
            self._smoke_hook = self.model.register_forward_hook(self._check_forward)
            self._check(self._frozen_error())

        def _frozen_error(self):
            required = {
                "embed_tokens.weight",
                "lm_head.weight",
                "verifier_lm_head.weight",
                "verifier_norm.weight",
            }
            found = set()
            optimized = {
                id(param)
                for opt in self.optimizers
                for group in opt.param_groups
                for param in group["params"]
            }
            for name, param in self.model.named_parameters():
                canonical = name.removeprefix("module.")
                if canonical in required:
                    found.add(canonical)
                    if (
                        param.requires_grad
                        or param.grad is not None
                        or id(param) in optimized
                    ):
                        return f"frozen target IO entered training: {canonical}"
            if found != required:
                return f"smoke cannot find frozen target IO: {sorted(required - found)}"
            return None

        def _check_forward(self, module, _inputs, output):
            loss = output[1]
            self._check(
                None
                if isinstance(loss, torch.Tensor)
                and loss.numel() == 1
                and torch.isfinite(loss).all().item()
                else "smoke observed a missing/non-finite scalar loss"
            )
            if module.training:
                self._smoke_train_calls += 1
            else:
                self._smoke_val_calls += 1

        def _optimizers_step(self):
            error = self._frozen_error()
            gradients = [
                param.grad
                for param in self.model.parameters()
                if param.requires_grad and param.grad is not None
            ]
            if not gradients or not all(
                torch.isfinite(g).all().item() for g in gradients
            ):
                error = "smoke gradients are missing or non-finite"
            elif not any(torch.count_nonzero(g).item() for g in gradients):
                error = "smoke observed only zero gradients"
            rates = [
                group["lr"] for opt in self.optimizers for group in opt.param_groups
            ]
            if not rates or not all(math.isfinite(lr) and lr > 0 for lr in rates):
                error = (
                    "smoke optimizer LR must be finite and positive before every update"
                )
            self._check(error)
            super()._optimizers_step()
            self._smoke_grad_steps += 1

        def val_epoch(self, epoch):
            metrics = super().val_epoch(epoch)
            valid = (
                metrics
                and "loss_epoch" in metrics
                and all(math.isfinite(float(value)) for value in metrics.values())
            )
            self._check(
                None if valid else "smoke validation metrics are empty/non-finite"
            )
            self._smoke_report["validation_metrics"] = metrics
            return metrics

        def run_training(self):
            try:
                super().run_training()
                expected = len(self.train_loader)
                valid = (
                    self._smoke_train_calls == expected
                    and self._smoke_grad_steps == expected
                    and self._smoke_val_calls == len(self.val_loader)
                    and self.global_step
                    == self._smoke_report["initial"]["global_step"] + expected
                )
                self._check(
                    None
                    if valid
                    else "smoke did not finish the required train/val steps"
                )
                self._check(self._frozen_error())
                self._check(
                    None
                    if all(
                        torch.isfinite(param).all().item()
                        for param in self.model.parameters()
                        if param.requires_grad
                    )
                    else "smoke optimizer produced non-finite parameters"
                )
                final = state_probe(self, torch)
                frozen_changed = [
                    name
                    for name, param in self.model.named_parameters()
                    if not param.requires_grad
                    and self._smoke_report["initial"]["model"][name]
                    != final["model"][name]
                ]
                self._check(
                    None
                    if not frozen_changed
                    else "smoke frozen parameter probes changed"
                )
                self._smoke_report.update(
                    status="passed",
                    final=final,
                    train_forward_calls=self._smoke_train_calls,
                    validation_forward_calls=self._smoke_val_calls,
                    gradient_steps=self._smoke_grad_steps,
                )
                settings.report_dir.mkdir(parents=True, exist_ok=True)
                path = settings.report_dir / f"{settings.phase}.rank-{self.rank}.json"
                # Exclusive create: never replace a previous run's acceptance evidence.
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(self._smoke_report, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            finally:
                self._smoke_hook.remove()

    return SmokeTrainer
