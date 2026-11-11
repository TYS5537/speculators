"""Loop HS is a persisted training policy, not checkpoint architecture."""

from types import SimpleNamespace

import pytest

from speculators.models.leagle import LEagleDraftModel
from speculators.train.config import TrainConfig
from speculators.train.legacy_cli import parse_train_args
from speculators.train.model_config import plan_pretrained_config_overrides
from tests.unit.models.test_leagle import make_model
from tests.unit.train.config.test_eagle_train_scripts import launch_args

FLAGS = [
    "--verifier-name-or-path",
    "dummy",
    "--speculator-type",
    "leagle",
    "--num-loops",
    "4",
    "--leagle-loop-hs-loss",
    "--leagle-loop-hs-loss-weight",
    "0.07",
]


def test_both_parsers_yaml_and_forward_kwargs(monkeypatch, tmp_path):
    config = TrainConfig.resolve(FLAGS)
    monkeypatch.setattr("sys.argv", ["train.py", *FLAGS])
    legacy = parse_train_args()
    assert legacy.leagle_loop_hs_loss == config.loss.leagle_loop_hs_loss is True
    assert (
        legacy.leagle_loop_hs_loss_weight
        == config.loss.leagle_loop_hs_loss_weight
        == 0.07
    )
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    restored = TrainConfig.resolve(["--config", str(path)])
    assert restored.loss == config.loss
    disabled = TrainConfig.resolve(["--config", str(path), "--no-leagle-loop-hs-loss"])
    assert not disabled.loss.leagle_loop_hs_loss
    training, validation = LEagleDraftModel.get_trainer_kwargs(**restored.flatten())
    for flags in (training, validation):
        assert flags["leagle_loop_hs_loss"] is True
        assert flags["leagle_loop_hs_loss_weight"] == 0.07
    checkpoint = make_model(loops=4).config
    for name in ("leagle_loop_hs_loss", "leagle_loop_hs_loss_weight"):
        assert name not in checkpoint.to_dict()
    _, overrides = plan_pretrained_config_overrides(
        SimpleNamespace(**restored.flatten()),
        checkpoint,
        {"leagle_loop_hs_loss", "leagle_loop_hs_loss_weight"},
    )
    assert overrides == {}


@pytest.mark.parametrize(
    "extra",
    [
        ["--speculator-type", "eagle3", "--num-loops", "1"],
        ["--candidate-head-only", "--from-pretrained", "draft"],
        ["--leagle-loop-hs-loss-weight", "-1"],
        ["--leagle-loop-hs-loss-weight", "nan"],
        ["--leagle-loop-hs-loss-weight", "inf"],
    ],
)
def test_invalid_flags_rejected_by_both_parsers(extra, monkeypatch):
    with pytest.raises(SystemExit):
        TrainConfig.resolve([*FLAGS, *extra])
    monkeypatch.setattr("sys.argv", ["train.py", *FLAGS, *extra])
    with pytest.raises(SystemExit):
        parse_train_args()


@pytest.mark.parametrize("master", ["--leagle", "--no-leagle"])
def test_optimization_masters_do_not_change_supervision(master):
    config = TrainConfig.resolve(
        [
            *FLAGS,
            "--draft-attn-impl",
            "sdpa",
            "--activation-checkpointing",
            "--fsdp-shard",
            f"{master}-safe-optimizations",
            f"{master}-numerical-optimizations",
        ]
    )
    assert config.loss.leagle_loop_hs_loss
    assert config.loss.leagle_loop_hs_loss_weight == 0.07


def test_script_new_run_and_resume_overrides(tmp_path):
    config = TrainConfig.resolve(
        launch_args(
            "leagle",
            MODEL="model",
            NUM_LOOPS="4",
            LEAGLE_LOOP_HS_LOSS="1",
            LEAGLE_LOOP_HS_LOSS_WEIGHT="0.03",
        )
    )
    assert config.loss.leagle_loop_hs_loss
    assert config.loss.leagle_loop_hs_loss_weight == 0.03
    saved = tmp_path / "checkpoints"
    saved.mkdir()
    (saved / "run.yaml").write_text(config.dump_yaml())
    for switch, weight, enabled, expected_weight in (
        ("", "", True, 0.03),
        ("0", "", False, 0.03),
        ("1", "0.02", True, 0.02),
    ):
        resumed = TrainConfig.resolve(
            launch_args(
                "leagle",
                RESUME="1",
                OUTPUT_DIR=tmp_path.as_posix(),
                LEAGLE_LOOP_HS_LOSS=switch,
                LEAGLE_LOOP_HS_LOSS_WEIGHT=weight,
            )
        )
        assert resumed.loss.leagle_loop_hs_loss is enabled
        assert resumed.loss.leagle_loop_hs_loss_weight == expected_weight
        assert resumed.draft.num_loops == 4
    defaults = TrainConfig.resolve(launch_args("leagle", MODEL="model"))
    assert not defaults.loss.leagle_loop_hs_loss
