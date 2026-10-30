"""DDP flags reach wrapping without entering checkpoint architecture configs."""

import copy
import inspect
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from speculators.models.leagle import LEagleDraftModel
from speculators.models.leagle.options import RUNTIME_DEFAULTS, TRAINING_DEFAULTS
from speculators.train import ddp_options
from speculators.train.config import TrainConfig
from speculators.train.leagle_optimizations import GROUP_DEFAULTS
from speculators.train.legacy_cli import parse_train_args
from speculators.train.trainer import Trainer
from tests.unit.models.test_leagle import make_model

from .test_activation_checkpointing_setup import _trainer
from .test_leagle_optimization_config import BASE


@pytest.mark.parametrize("legacy", [False, True])
def test_ddp_flags_are_opt_in_typed_and_restricted_to_ddp_leagle(legacy):
    def parse(extra):
        argv = [*BASE, *extra]
        return (
            vars(parse_train_args(argv))
            if legacy
            else TrainConfig.resolve(argv).flatten()
        )

    defaults = parse([])
    assert not any(defaults[name] for name in ddp_options.DDP_FLAGS)
    assert defaults[ddp_options.BUCKET_CAP] is None
    flags = ["--" + name.replace("_", "-") for name in ddp_options.DDP_FLAGS]
    flags += ["--leagle-ddp-bucket-cap-mb", "64"]
    enabled = parse(flags)
    assert all(enabled[name] for name in ddp_options.DDP_FLAGS)
    assert enabled[ddp_options.BUCKET_CAP] == 64
    train, val = LEagleDraftModel.get_trainer_kwargs(**enabled)
    assert not set(RUNTIME_DEFAULTS).intersection(train)
    assert not set(RUNTIME_DEFAULTS).intersection(val)
    for name in ddp_options.DDP_FLAGS:
        flag = "--" + name.replace("_", "-")
        for extra in ([flag, "--fsdp-shard"], [flag, "--speculator-type", "eagle3"]):
            with pytest.raises(SystemExit):
                parse(extra)
    for extra in (["--fsdp-shard"], ["--speculator-type", "eagle3"]):
        with pytest.raises(SystemExit):
            parse(["--leagle-ddp-bucket-cap-mb", "64", *extra])
    for value in ("0", "-1", "1.5", "true"):
        with pytest.raises(SystemExit):
            parse(["--leagle-ddp-bucket-cap-mb", value])


@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("numerical", [False, True])
@pytest.mark.parametrize("fsdp", [False, True])
def test_masters_select_only_the_current_distributed_strategy(safe, numerical, fsdp):
    config = TrainConfig.resolve(
        [
            *BASE,
            "--fsdp-shard" if fsdp else "--no-fsdp-shard",
            "--leagle-safe-optimizations" if safe else "--no-leagle-safe-optimizations",
            "--leagle-numerical-optimizations"
            if numerical
            else "--no-leagle-numerical-optimizations",
        ]
    )
    assert config.trainer.leagle_ddp_gradient_as_bucket_view is (safe and not fsdp)
    assert config.trainer.leagle_ddp_batched_grad_copy is (numerical and not fsdp)
    assert config.trainer.leagle_fsdp_keep_forward_params is (safe and fsdp)
    assert config.trainer.leagle_ddp_bucket_cap_mb is None


def test_ddp_options_roundtrip_and_numerical_off_resets_bucket_policy(tmp_path):
    config = TrainConfig.resolve(
        [
            *BASE,
            "--leagle-ddp-bucket-cap-mb",
            "64",
            "--leagle-safe-optimizations",
            "--leagle-numerical-optimizations",
        ]
    )
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    assert TrainConfig.resolve(["--config", str(path)]).flatten() == config.flatten()
    off = TrainConfig.resolve(
        [
            "--config",
            str(path),
            "--no-leagle-numerical-optimizations",
            "--leagle-ddp-bucket-cap-mb",
            "100",
            "--leagle-ddp-batched-grad-copy",
        ]
    )
    assert off.trainer.leagle_ddp_gradient_as_bucket_view
    assert not off.trainer.leagle_ddp_batched_grad_copy
    assert off.trainer.leagle_ddp_bucket_cap_mb is None


def old_ddp(module, gradient_as_bucket_view=False, bucket_cap_mb=None):
    return module


def new_ddp(
    module, gradient_as_bucket_view=False, batched_grad_copy=False, bucket_cap_mb=None
):
    return module


def test_runtime_fallback_logs_and_records_effective_settings(
    monkeypatch, caplog, tmp_path
):
    config = TrainConfig.resolve(
        [
            *BASE,
            "--leagle-ddp-gradient-as-bucket-view",
            "--leagle-ddp-batched-grad-copy",
            "--leagle-ddp-bucket-cap-mb",
            "64",
        ]
    )
    args = Namespace(**config.flatten())
    monkeypatch.setattr(ddp_options, "DistributedDataParallel", old_ddp)
    assert ddp_options.apply_runtime_ddp_options(config, args)
    assert not args.leagle_ddp_batched_grad_copy
    assert not config.trainer.leagle_ddp_batched_grad_copy
    assert config.provenance["leagle_ddp_batched_grad_copy"] == "runtime"
    assert "does not support" in caplog.text
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    restored = TrainConfig.resolve(["--config", str(path)])
    assert not restored.trainer.leagle_ddp_batched_grad_copy
    options, updates = ddp_options.resolve_ddp_options(args)
    assert options == {"gradient_as_bucket_view": True, "bucket_cap_mb": 64}
    assert not updates


def test_single_process_skips_ddp_and_defaults_preserve_native_bucket_policy():
    options, updates = ddp_options.resolve_ddp_options(
        SimpleNamespace(), constructor=new_ddp
    )
    assert options == updates == {}
    selected = SimpleNamespace(
        **dict.fromkeys(ddp_options.DDP_FLAGS, True), leagle_ddp_bucket_cap_mb=64
    )
    options, updates = ddp_options.resolve_ddp_options(selected, constructor=new_ddp)
    assert options == {
        "gradient_as_bucket_view": True,
        "batched_grad_copy": True,
        "bucket_cap_mb": 64,
    }
    assert updates == {}
    options, updates = ddp_options.resolve_ddp_options(selected, distributed=False)
    assert not options
    assert updates == {
        **dict.fromkeys(ddp_options.DDP_FLAGS, False),
        ddp_options.BUCKET_CAP: None,
    }


@pytest.mark.parametrize("supported", [False, True])
def test_production_trainer_passes_only_supported_requested_ddp_options(
    monkeypatch, supported
):
    model = make_model(layers=1, loops=1)
    trainer = _trainer(model, strategy="ddp", resume=True)
    trainer.config = trainer.config._replace(
        leagle_ddp_gradient_as_bucket_view=True,
        leagle_ddp_batched_grad_copy=True,
        leagle_ddp_bucket_cap_mb=64,
    )
    seen = []

    def wrap(module, **kwargs):
        seen.append(kwargs)
        return module

    wrap.__signature__ = inspect.signature(new_ddp if supported else old_ddp)
    monkeypatch.setattr("speculators.train.trainer.DistributedDataParallel", wrap)
    monkeypatch.setattr("speculators.train.trainer.dist.get_rank", lambda: 0)
    Trainer._setup_model_ddp(trainer, load_checkpoint=True)
    expected = {"gradient_as_bucket_view": True, "bucket_cap_mb": 64}
    if supported:
        expected["batched_grad_copy"] = True
    assert seen == [expected]
    assert trainer.config.leagle_ddp_batched_grad_copy is supported
    trainer.checkpointer.load_model_state_dict.assert_called_once_with(model)


def test_all_execution_options_stay_out_of_saved_model_config(tmp_path):
    model = make_model(layers=1, loops=2, attention="sdpa")
    before = copy.deepcopy(model.config.model_dump(mode="json"))
    config = TrainConfig.resolve(
        [
            *BASE,
            "--leagle-safe-optimizations",
            "--leagle-numerical-optimizations",
            "--leagle-ddp-bucket-cap-mb",
            "64",
        ]
    )
    model.get_trainer_kwargs(**config.flatten())
    model.share_output_head()
    model.set_activation_checkpointing(True)
    assert model.config.model_dump(mode="json") == before
    model.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    forbidden = set(TRAINING_DEFAULTS) | set(RUNTIME_DEFAULTS) | set(GROUP_DEFAULTS)

    def check(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)

    check(saved)
    # Model topology and the checkpoint's parameter names/shapes stay unchanged.
    restored = LEagleDraftModel.from_pretrained(tmp_path)
    expected = {
        k: v
        for k, v in model.state_dict().items()
        if k not in model._keys_to_ignore_on_save
    }
    actual = {k: v for k, v in restored.state_dict().items() if k in expected}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
