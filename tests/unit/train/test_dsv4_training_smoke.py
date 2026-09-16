"""CPU checks of smoke assertions, not evidence of A3 model/kernel support."""

import copy
import json
from pathlib import Path
from typing import NamedTuple

import pytest

from speculators_dsv4.training_smoke import (
    BoundedLoader,
    SmokeSettings,
    make_smoke_trainer,
    recipe_digest,
    verify_restored_state,
)

torch = pytest.importorskip("torch")


class Config(NamedTuple):
    save_path: str
    optimizer: str = "muon"
    scheduler_type: str = "linear"
    fsdp_shard: bool = False
    num_epochs: int = 5
    resume_from_checkpoint: bool = True
    scheduler_total_steps: int | None = None
    scheduler_warmup_steps: int | None = None
    scheduler_warmup_ratio: float | None = None
    checkpoint_freq: int = 1
    save_best: bool = True
    log_freq: int = 1


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        for name in ("embed_tokens", "lm_head", "verifier_lm_head", "verifier_norm"):
            layer = torch.nn.Linear(2, 2, bias=False)
            layer.weight.requires_grad_(False)
            self.add_module(name, layer)

    def forward(self, inputs):
        loss = self.proj(inputs).square().mean()
        return None, loss, {"loss": loss.detach().item()}


class TinyBase:
    """Real autograd/optimizer, tiny fixture loop (NOT the production trainer)."""

    def __init__(self, model, config, train_loader, val_loader):
        self.model, self.config, self.rank = model, config, 0
        self.train_loader, self.val_loader = train_loader, val_loader
        self.optimizers = [torch.optim.AdamW(model.proj.parameters(), lr=0.02)]
        total = config.scheduler_total_steps
        self.schedulers = [
            torch.optim.lr_scheduler.LambdaLR(
                self.optimizers[0], lambda step: 1 - step / total
            )
        ]
        self.current_epoch, self.global_step = 0, 0
        if config.resume_from_checkpoint:
            saved = torch.load(
                Path(config.save_path) / "0" / "tiny.pt", weights_only=True
            )
            model.load_state_dict(saved["model"])
            self.optimizers[0].load_state_dict(saved["optimizer"])
            self.schedulers[0].load_state_dict(saved["scheduler"])
            self.current_epoch, self.global_step = 1, saved["step"]

    def _optimizers_step(self):
        self.optimizers[0].step()

    def val_epoch(self, _epoch):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                losses.append(self.model(batch)[1].item())
        return {"loss_epoch": sum(losses) / len(losses)}

    def run_training(self):
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.model.train()
            for batch in self.train_loader:
                self.optimizers[0].zero_grad()
                self.model(batch)[1].backward()
                self._optimizers_step()
                self.schedulers[0].step()
                self.global_step += 1
            path = Path(self.config.save_path) / str(epoch)
            path.mkdir(parents=True)
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizers[0].state_dict(),
                    "scheduler": self.schedulers[0].state_dict(),
                    "step": self.global_step,
                },
                path / "tiny.pt",
            )
            for name in (
                "config.json",
                "model.safetensors",
                "optimizer_state_dict.pt",
                "scheduler_state_dict.pt",
                "training_state.json",
            ):
                (path / name).touch()
            self.val_epoch(epoch)


def _trainer(tmp_path, phase="fresh", *, model=None, config=None, base=TinyBase):
    torch.manual_seed(7)
    settings = SmokeSettings(phase, tmp_path / "reports", recipe_sha256="recipe")
    config = config or Config(str(tmp_path / "checkpoints"))
    loader = [torch.ones(2, 2)] * 4
    cls = make_smoke_trainer(base, settings)
    return cls(model or TinyModel(), config, loader, loader)


def test_two_phase_checks_and_reports(tmp_path):
    first = _trainer(tmp_path)
    assert first.config.scheduler_total_steps == 4
    assert first.config.num_epochs == 1
    assert not first.config.save_best
    first.run_training()
    second = _trainer(tmp_path, "resume")
    assert second.global_step == 2
    assert second.optimizers[0].param_groups[0]["lr"] == 0.01
    second.run_training()
    for phase in ("fresh", "resume"):
        report = json.loads((tmp_path / "reports" / f"{phase}.rank-0.json").read_text())
        assert report["status"] == "passed"
        assert report["gradient_steps"] == 2
        assert report["validation_forward_calls"] == 1
        assert report["device"] == "cpu"
    assert second.global_step == 4


def test_bounded_loader_keeps_sampler():
    class Loader(list):
        batch_sampler = object()

    source = Loader([1, 2, 3])
    capped = BoundedLoader(source, 2)
    assert len(capped) == 2
    assert list(capped) == [1, 2]
    assert capped.batch_sampler is source.batch_sampler


@pytest.mark.parametrize(("loader", "limit"), [(None, 1), ([], 1), ([1], 0)])
def test_empty_loaders_rejected(loader, limit):
    with pytest.raises(ValueError, match="nonempty"):
        BoundedLoader(loader, limit)


@pytest.mark.parametrize("key", ["model", "optimizers", "schedulers", "global_step"])
def test_any_restore_probe_mismatch_rejected(key):
    saved = {"model": {}, "optimizers": [], "schedulers": [], "global_step": 2}
    current = copy.deepcopy(saved)
    current[key] = "wrong"
    with pytest.raises(ValueError, match=key):
        verify_restored_state(saved, current)


def test_refuses_existing_report_or_checkpoint(tmp_path):
    _trainer(tmp_path).run_training()
    with pytest.raises(ValueError, match="report already exists"):
        _trainer(tmp_path)
    (tmp_path / "reports" / "fresh.rank-0.json").unlink()
    with pytest.raises(ValueError, match="existing training checkpoint"):
        _trainer(tmp_path)


def test_resume_requires_complete_fresh_checkpoint(tmp_path):
    _trainer(tmp_path).run_training()
    (tmp_path / "checkpoints" / "0" / "scheduler_state_dict.pt").unlink()
    with pytest.raises(ValueError, match="missing scheduler"):
        _trainer(tmp_path, "resume")


def test_changed_recipe_rejected(tmp_path):
    _trainer(tmp_path).run_training()
    settings = SmokeSettings("resume", tmp_path / "reports", recipe_sha256="changed")
    cls = make_smoke_trainer(TinyBase, settings)
    with pytest.raises(ValueError, match="recipe arguments changed"):
        cls(TinyModel(), Config(str(tmp_path / "checkpoints")), [1], [1])


@pytest.mark.parametrize("invalid", ["zero", "nan", "frozen", "lr"])
def test_invalid_optimizer_step_fails(tmp_path, invalid):
    trainer = _trainer(tmp_path)
    for param in trainer.model.proj.parameters():
        param.grad = torch.ones_like(param)
    if invalid == "zero":
        for param in trainer.model.proj.parameters():
            param.grad.zero_()
    elif invalid == "nan":
        trainer.model.proj.weight.grad.fill_(float("nan"))
    elif invalid == "frozen":
        trainer.model.verifier_norm.weight.grad = torch.ones(2, 2)
    else:
        trainer.optimizers[0].param_groups[0]["lr"] = 0
    with pytest.raises(ValueError, match="smoke|frozen"):
        trainer._optimizers_step()
    assert not (tmp_path / "reports").exists()


def test_nonfinite_forward_fails_without_pass_report(tmp_path):
    trainer = _trainer(tmp_path)
    with torch.no_grad():
        trainer.model.proj.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="non-finite scalar loss"):
        trainer.run_training()
    assert not (tmp_path / "reports").exists()


def test_digest_binds_argument_order_and_values():
    assert recipe_digest(["--lr", "0.01"]) != recipe_digest(["--lr", "0.02"])


@pytest.mark.parametrize(
    ("phase", "train", "val"), [("bad", 1, 1), ("fresh", 0, 1), ("resume", 1, 0)]
)
def test_invalid_settings(phase, train, val):
    with pytest.raises(ValueError):
        SmokeSettings(phase, Path("reports"), train, val)
