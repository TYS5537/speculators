"""Checkpoint-based candidate training keeps source/serving contracts intact."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from speculators import SpeculatorModelConfig
from speculators.models.leagle import LEagleDraftModel
from speculators.models.leagle.svd_head import svd_factors
from speculators.train.config import TrainConfig
from speculators.train.leagle_candidate import validate_candidate_destination
from speculators.train.leagle_optimizations import apply_runtime_optimization_groups
from speculators.train.legacy_cli import parse_train_args
from speculators.train.model_init import build_draft_model
from speculators_eval import leagle
from tests.unit.evaluate.test_leagle_offline_eval import save_fixture


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(723)
    torch.set_num_threads(1)


def flags(source="unused", target="unused"):
    return [
        "--verifier-name-or-path",
        str(target),
        "--speculator-type",
        "leagle",
        "--from-pretrained",
        str(source),
        "--candidate-head-only",
        "--candidate-head-rank",
        "4",
        "--candidate-head-top-k",
        "3",
        "--candidate-head-chunk-size",
        "2",
        "--draft-attn-impl",
        "sdpa",
    ]


def load_stage_two(config):
    args = argparse.Namespace(**config.flatten())
    args._provided_model_config_dests = {
        name for name, origin in config.provenance.items() if origin != "default"
    }
    model = build_draft_model(args, LEagleDraftModel, None, None, None)
    apply_runtime_optimization_groups(config, args, model)
    return model, args


@pytest.mark.parametrize("initialization", ["svd", "random"])
@pytest.mark.parametrize(
    "masters", [[], ["--leagle-safe-optimizations", "--leagle-numerical-optimizations"]]
)
def test_cli_yaml_policy_and_optimization_resolution(initialization, masters, tmp_path):
    argv = [*flags(), "--candidate-head-init", initialization, *masters]
    cfg = TrainConfig.resolve(argv)
    legacy = parse_train_args(argv)
    assert cfg.draft.candidate_head_only
    assert legacy.candidate_head_only
    assert cfg.draft.candidate_head_init == legacy.candidate_head_init == initialization
    for name in (
        "leagle_teacher_cache",
        "leagle_stream_main_loss",
        "leagle_chunked_loss",
        "leagle_masked_head",
        "leagle_shared_loss_normalization",
        "leagle_selective_checkpoint",
        "leagle_align_candidate_chunks",
    ):
        assert not cfg.flatten()[name]
        assert not getattr(legacy, name)
    path = tmp_path / "run.yaml"
    path.write_text(cfg.dump_yaml())
    restored = TrainConfig.resolve(["--config", str(path)])
    assert restored.flatten() == cfg.flatten()


@pytest.mark.parametrize(
    "extra",
    [
        ["--speculator-type", "eagle3"],
        ["--from-pretrained", ""],
        ["--no-candidate-rerank"],
        ["--candidate-head-init", "other"],
        ["--draft-vocab-size", "12"],
        ["--t2d-path", "mapping.npy"],
    ],
)
def test_invalid_policies_fail_in_both_parsers(extra):
    for parse in (TrainConfig.resolve, parse_train_args):
        with pytest.raises(SystemExit):
            parse([*flags(), *extra])


def test_load_train_save_reload_and_evaluate_existing_draft(tmp_path):
    evaluation = save_fixture(tmp_path)
    source = Path(evaluation.draft_model)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).digest()
        for path in source.iterdir()
        if path.is_file()
    }
    cfg = TrainConfig.resolve(flags(source, evaluation.verifier_model))
    model, args = load_stage_two(cfg)
    assert model.candidate_head_only
    assert model.config.num_loops == 3
    assert len(model.layers) == 2
    assert model.config.aux_hidden_state_layer_ids == [1, 2]
    assert cfg.draft.candidate_head
    assert cfg.draft.candidate_rerank
    assert cfg.draft.candidate_head_rank == 4
    up, down = svd_factors(model.lm_head.weight, 4)
    torch.testing.assert_close(model.candidate_head.up_proj.weight, up, rtol=0, atol=0)
    torch.testing.assert_close(
        model.candidate_head.down_proj.weight, down, rtol=0, atol=0
    )
    # Use the real frozen model and a real optimizer, then serve the saved output.
    original = copy.deepcopy(model.state_dict())
    inputs = {
        "hidden_states": torch.randn(1, 6, 32),
        "input_ids": torch.tensor([[5, 9, 2, 7, 4, 3]]),
        "document_ids": torch.zeros(1, 6, dtype=torch.long),
        "loss_mask": torch.ones(1, 6),
        "position_ids": torch.arange(1, 7).unsqueeze(0),
    }
    train, validation = model.get_trainer_kwargs(**vars(args))
    assert validation["candidate_head_train_stride"] == 1
    model.train()
    optimizer = torch.optim.AdamW(model.candidate_head.parameters(), lr=0.001)
    _, loss, _ = model(**inputs, **train)
    loss.backward()
    optimizer.step()
    for name, value in original.items():
        if not name.startswith("candidate_head."):
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    output = tmp_path / "stage-two"
    model.save_pretrained(output)
    saved = json.loads((output / "config.json").read_text())
    assert saved["candidate_head"]
    assert saved["candidate_rerank"]
    assert "candidate_head_only" not in saved
    assert "candidate_head_init" not in saved
    assert before == {
        path.name: hashlib.sha256(path.read_bytes()).digest()
        for path in source.iterdir()
        if path.is_file()
    }
    # Continued head-only training loads learned factors, not a fresh SVD.
    continued, _ = load_stage_two(
        TrainConfig.resolve(flags(output, evaluation.verifier_model))
    )
    torch.testing.assert_close(
        continued.state_dict(), model.state_dict(), rtol=0, atol=0
    )
    assert continued.candidate_head_only
    # The saved run retains the second-stage policy independently of model config.
    run = tmp_path / "run.yaml"
    run.write_text(cfg.dump_yaml())
    restored_cfg = TrainConfig.resolve(["--config", str(run)])
    fresh, _ = load_stage_two(restored_cfg)
    fresh.load_state_dict(model.state_dict())  # Trainer checkpoint restoration.
    torch.testing.assert_close(
        fresh.state_dict(), continued.state_dict(), rtol=0, atol=0
    )
    assert all(
        not p.requires_grad
        for name, p in fresh.named_parameters()
        if not name.startswith("candidate_head.")
    )
    evaluation.draft_model = str(output)
    leagle.run(evaluation, entrypoint=Path("scripts/evaluate/leagle_offline_eval.py"))
    rows = json.loads((evaluation.output_dir / "summary.json").read_text())
    assert rows[0]["num_requests"] == 2
    assert 1 <= rows[0]["acceptance_length"] <= 8


def test_second_stage_rejects_config_only_or_incompatible_checkpoints(tmp_path):
    evaluation = save_fixture(tmp_path)
    source = Path(evaluation.draft_model)
    saved = SpeculatorModelConfig.from_pretrained(source)
    config_only = tmp_path / "config-only"
    saved.save_pretrained(config_only)
    with pytest.raises(ValueError, match="trained checkpoint weights"):
        load_stage_two(
            TrainConfig.resolve(flags(config_only, evaluation.verifier_model))
        )
    with pytest.raises(ValueError, match="cannot disable"):
        load_stage_two(
            TrainConfig.resolve(
                [*flags(source, evaluation.verifier_model), "--no-candidate-head"]
            )
        )
    # Normal pretrained training still rejects adding a candidate head implicitly.
    normal = TrainConfig.resolve(
        [
            *flags(source, evaluation.verifier_model),
            "--no-candidate-head-only",
            "--candidate-head",
        ]
    )
    with pytest.raises(ValueError, match="cannot change checkpoint"):
        load_stage_two(normal)
    model, _ = load_stage_two(
        TrainConfig.resolve(flags(source, evaluation.verifier_model))
    )
    model.config.candidate_rerank = False
    incompatible = tmp_path / "low-rank-main"
    model.save_pretrained(incompatible)
    with pytest.raises(ValueError, match="candidate reranking"):
        load_stage_two(
            TrainConfig.resolve(flags(incompatible, evaluation.verifier_model))
        )
    # A corrupt existing head remains an error, not permission to reinitialize it.
    model.config.candidate_rerank = True
    model.save_pretrained(incompatible)
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    path = incompatible / "model.safetensors"
    weights = load_file(path)
    weights.pop("candidate_head.down_proj.weight")
    save_file(weights, path)
    with pytest.raises(ValueError, match="missing compatible candidate"):
        load_stage_two(
            TrainConfig.resolve(flags(incompatible, evaluation.verifier_model))
        )


def test_destination_guard_preserves_source_tree(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for destination in (source, source / "child", tmp_path):
        with pytest.raises(ValueError, match="separate save directory"):
            validate_candidate_destination(source, destination)
    validate_candidate_destination(source, tmp_path / "new-run")


def test_missing_frozen_backbone_weights_are_rejected(tmp_path):
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    evaluation = save_fixture(tmp_path)
    path = Path(evaluation.draft_model) / "model.safetensors"
    weights = load_file(path)
    weights.pop("input_fc.weight")
    save_file(weights, path)
    with pytest.raises(ValueError, match="Source draft checkpoint"):
        load_stage_two(
            TrainConfig.resolve(
                flags(evaluation.draft_model, evaluation.verifier_model)
            )
        )
