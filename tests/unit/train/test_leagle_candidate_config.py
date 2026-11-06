"""New-run candidate knobs reach construction, checkpoints and both CLI parsers."""

import copy
from types import SimpleNamespace

import pytest
import torch

from speculators.models.leagle import LEagleDraftModel
from speculators.train.config import TrainConfig
from speculators.train.legacy_cli import parse_train_args
from speculators.train.model_config import plan_pretrained_config_overrides
from speculators.train.optimizers import split_named_params_for_muon
from tests.unit.models.test_leagle import make_model

FLAGS = [
    "--verifier-name-or-path",
    "dummy",
    "--speculator-type",
    "leagle",
    "--candidate-head",
    "--candidate-head-rank",
    "4",
    "--candidate-head-top-k",
    "3",
    "--candidate-head-loss-weight",
    "0.2",
    "--candidate-head-chunk-size",
    "2",
]


@pytest.mark.parametrize("rerank", [True, False])
def test_cli_yaml_round_trip_and_legacy_parser_match(monkeypatch, tmp_path, rerank):
    flags = [*FLAGS, "--candidate-rerank" if rerank else "--no-candidate-rerank"]
    current = TrainConfig.resolve(flags)
    monkeypatch.setattr("sys.argv", ["train.py", *flags])
    legacy = parse_train_args()
    for name in (
        "candidate_head",
        "candidate_rerank",
        "candidate_head_rank",
        "candidate_head_top_k",
        "candidate_head_loss_weight",
        "candidate_head_chunk_size",
        "candidate_head_train_stride",
    ):
        assert current.flatten()[name] == getattr(legacy, name)
    path = tmp_path / "run.yaml"
    path.write_text(current.dump_yaml())
    restored = TrainConfig.resolve(["--config", str(path)])
    assert restored.draft == current.draft
    disabled = TrainConfig.resolve(["--config", str(path), "--no-candidate-head"])
    assert not disabled.draft.candidate_head


@pytest.mark.parametrize(
    "extra",
    [
        ["--speculator-type", "eagle3"],
        ["--candidate-head-rank", "0"],
        ["--candidate-head-top-k", "0"],
        ["--candidate-head-chunk-size", "0"],
        ["--candidate-head-train-stride", "0"],
        ["--candidate-head-train-stride", "1.5"],
        ["--candidate-head-loss-weight", "nan"],
        ["--candidate-head-loss-weight", "-1"],
    ],
)
def test_invalid_candidate_training_flags_fail_in_both_parsers(extra, monkeypatch):
    with pytest.raises(SystemExit):
        TrainConfig.resolve([*FLAGS, *extra])
    monkeypatch.setattr("sys.argv", ["train.py", *FLAGS, *extra])
    with pytest.raises(SystemExit):
        parse_train_args()


@pytest.mark.parametrize("rerank", [True, False])
def test_from_training_args_builds_trainable_head_without_svd_or_conversion(
    monkeypatch,
    rerank,
):
    seed_model = make_model()
    decoder = copy.deepcopy(seed_model.config.transformer_layer_config)
    monkeypatch.setattr(
        "speculators.models.leagle.core.resolve_target_layer_ids",
        lambda *args, **kwargs: [0, 1],
    )
    monkeypatch.setattr(
        "speculators.models.leagle.core.VerifierConfig.from_pretrained",
        lambda *args, **kwargs: seed_model.config.speculators_config.verifier,
    )

    def load_target(model):
        with torch.no_grad():
            model.embed_tokens.weight.copy_(seed_model.embed_tokens.weight)
            model.lm_head.weight.copy_(seed_model.lm_head.weight)
            model.verifier_lm_head.weight.copy_(seed_model.verifier_lm_head.weight)

    monkeypatch.setattr(LEagleDraftModel, "load_verifier_weights", load_target)
    flat = TrainConfig.resolve(FLAGS).flatten()
    flat["candidate_rerank"] = rerank
    flat["draft_vocab_size"] = 24
    model = LEagleDraftModel.from_training_args(verifier_config=decoder, **flat)
    assert model.config.candidate_head
    assert model.low_rank_main is (not rerank)
    assert model.candidate_rerank is rerank
    assert model.config.candidate_head_loss_weight == 0.2
    assert model.config.candidate_head_chunk_size == 2
    assert all(
        p.requires_grad and torch.isfinite(p).all()
        for p in model.candidate_head.parameters()
    )
    assert not model.lm_head.weight.requires_grad


def test_from_pretrained_preserves_or_rejects_architecture_overrides():
    config = make_model(candidate_head=True).config
    defaults = SimpleNamespace(**TrainConfig(speculator_type="leagle").flatten())
    inherited, changes = plan_pretrained_config_overrides(defaults, config, set())
    assert inherited["candidate_head"] is True
    assert inherited["candidate_head_rank"] == 4
    assert changes == {}
    for name, value in (
        ("candidate_head", False),
        ("candidate_head_rank", 2),
        ("candidate_rerank", False),
    ):
        defaults.__dict__[name] = value
        with pytest.raises(ValueError, match="cannot change checkpoint"):
            plan_pretrained_config_overrides(defaults, config, {name})


def test_candidate_stride_is_training_only_and_cli_yaml_restore_it(
    tmp_path, monkeypatch
):
    flags = [*FLAGS, "--candidate-head-train-stride", "8"]
    config = TrainConfig.resolve(flags)
    monkeypatch.setattr("sys.argv", ["train.py", *flags])
    assert parse_train_args().candidate_head_train_stride == 8
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    restored = TrainConfig.resolve(["--config", str(path)])
    assert restored.draft.candidate_head_train_stride == 8
    model = make_model(candidate_head=True)
    assert "candidate_head_train_stride" not in model.config.to_dict()
    args = SimpleNamespace(**restored.flatten())
    _, overrides = plan_pretrained_config_overrides(
        args, model.config, {"candidate_head_train_stride"}
    )
    assert not overrides


@pytest.mark.parametrize("recipe", ["upstream", "legacy"])
def test_low_rank_output_factors_use_adamw_with_muon(recipe):
    model = make_model(candidate_head=True)
    muon, adam = split_named_params_for_muon(model, training_recipe=recipe)
    assert not any(name.startswith("candidate_head.") for name, _ in muon)
    assert {name for name, _ in adam if name.startswith("candidate_head.")} == {
        "candidate_head.down_proj.weight",
        "candidate_head.up_proj.weight",
    }
