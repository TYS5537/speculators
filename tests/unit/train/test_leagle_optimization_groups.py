"""Master controls cannot silently enable numerical paths through dependencies."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import yaml

from speculators.models.leagle import LEagleDraftModel
from speculators.models.leagle.options import RUNTIME_DEFAULTS, TRAINING_DEFAULTS
from speculators.train.config import TrainConfig
from speculators.train.leagle_optimizations import (
    GROUP_DEFAULTS,
    NUMERICAL_OPTIONS,
    SAFE_OPTIONS,
    apply_runtime_optimization_groups,
    resolve_optimization_groups,
)
from speculators.train.legacy_cli import parse_train_args
from tests.unit.models.test_leagle import make_model
from tests.unit.models.test_leagle_training_diagnostics import packed_batch

BASE = [
    "--verifier-name-or-path",
    "dummy",
    "--speculator-type",
    "leagle",
    "--draft-attn-impl",
    "sdpa",
]


def typed(argv):
    return TrainConfig.resolve(argv).flatten()


def legacy(argv):
    return vars(parse_train_args(argv))


@pytest.mark.parametrize("parse", [typed, legacy])
def test_unset_masters_preserve_individual_controls(parse):
    result = parse([*BASE, "--leagle-chunked-loss", "--leagle-teacher-cache"])
    assert all(result[name] is None for name in GROUP_DEFAULTS)
    assert result["leagle_chunked_loss"]
    assert result["leagle_teacher_cache"]
    assert not result["leagle_sparse_attention"]
    assert not resolve_optimization_groups(result)


def test_groups_cover_every_execution_boolean_once():
    covered = [*SAFE_OPTIONS, *NUMERICAL_OPTIONS, "leagle_candidate_backward_recompute"]
    booleans = {
        name
        for name, value in {**TRAINING_DEFAULTS, **RUNTIME_DEFAULTS}.items()
        if isinstance(value, bool)
    }
    assert len(covered) == len(set(covered))
    assert set(covered) == booleans


@pytest.mark.parametrize("parse", [typed, legacy])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("safe", [False, True])
def test_numerical_off_overrides_all_members_and_implicit_dependencies(
    parse, candidate, safe
):
    # Even explicit member flags must not bypass the master's off setting.
    result = parse(
        [
            *BASE,
            "--fsdp-shard",
            "--activation-checkpointing",
            *(
                "--" + name.replace("_", "-")
                for name in (*NUMERICAL_OPTIONS, *SAFE_OPTIONS)
            ),
            "--leagle-safe-optimizations" if safe else "--no-leagle-safe-optimizations",
            "--no-leagle-numerical-optimizations",
            "--candidate-head" if candidate else "--no-candidate-head",
        ]
    )
    assert not any(result[name] for name in NUMERICAL_OPTIONS)
    assert not result["leagle_teacher_cache"]
    assert not result["leagle_teacher_cache_supervised"]
    assert not result["leagle_cache_frozen_head_weights"]
    assert not result["leagle_selective_checkpoint"]
    assert not result["leagle_cache_attention_masks"]
    assert result["leagle_share_output_head"] is safe
    assert result["leagle_fsdp_keep_forward_params"] is safe
    assert result["leagle_candidate_backward_recompute"] is candidate
    assert result["candidate_head"] is candidate


@pytest.mark.parametrize("parse", [typed, legacy])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("safe", [False, True])
def test_numerical_on_expands_only_existing_capabilities(parse, candidate, safe):
    result = parse(
        [
            *BASE,
            "--fsdp-shard",
            "--activation-checkpointing",
            "--leagle-numerical-optimizations",
            "--no-leagle-reuse-training-kv",
            "--leagle-safe-optimizations" if safe else "--no-leagle-safe-optimizations",
            "--candidate-head" if candidate else "--no-candidate-head",
        ]
    )
    assert result["leagle_reuse_training_kv"]
    assert result["leagle_sparse_attention"]
    assert result["leagle_chunked_loss"]
    assert result["leagle_masked_head"]
    assert result["leagle_candidate_grad_addmm"] is candidate
    assert result["leagle_align_candidate_chunks"] is candidate
    assert not result["leagle_candidate_backward_recompute"]
    assert all(
        result[name] is safe
        for name in SAFE_OPTIONS
        if name != "leagle_ddp_gradient_as_bucket_view"
    )
    assert not result["leagle_ddp_gradient_as_bucket_view"]
    assert not result["leagle_ddp_batched_grad_copy"]
    assert result["candidate_head"] is candidate


@pytest.mark.parametrize("parse", [typed, legacy])
def test_safe_alone_never_enables_chunking_or_sparse_attention(parse):
    result = parse([*BASE, "--leagle-safe-optimizations"])
    assert not any(result[name] for name in NUMERICAL_OPTIONS)
    assert not result["leagle_teacher_cache"]
    assert not result["leagle_cache_frozen_head_weights"]
    assert not result["leagle_cache_teacher_indices"]
    assert not result["leagle_cache_attention_masks"]
    assert not result["leagle_selective_checkpoint"]
    assert not result["leagle_fsdp_keep_forward_params"]
    assert result["leagle_share_output_head"]
    assert result["leagle_ddp_gradient_as_bucket_view"]


@pytest.mark.parametrize("parse", [typed, legacy])
def test_safe_uses_individually_selected_prerequisites(parse):
    result = parse(
        [
            *BASE,
            "--leagle-safe-optimizations",
            "--leagle-chunked-loss",
            "--leagle-sparse-attention",
        ]
    )
    assert result["leagle_teacher_cache"]
    assert result["leagle_teacher_cache_supervised"]
    assert result["leagle_cache_frozen_head_weights"]
    assert result["leagle_cache_teacher_indices"]
    assert result["leagle_cache_attention_masks"]
    assert not result["leagle_selective_checkpoint"]  # No activation checkpointing.


@pytest.mark.parametrize("parse", [typed, legacy])
@pytest.mark.parametrize(
    "flag",
    [
        "--leagle-safe-optimizations",
        "--no-leagle-safe-optimizations",
        "--leagle-numerical-optimizations",
        "--no-leagle-numerical-optimizations",
    ],
)
def test_groups_reject_other_algorithms(parse, flag):
    with pytest.raises(SystemExit):
        parse([*BASE, "--speculator-type", "eagle3", flag])


@pytest.mark.parametrize("parse", [typed, legacy])
def test_numerical_group_keeps_attention_backend_validation(parse):
    with pytest.raises(SystemExit):
        parse([*BASE, "--leagle-numerical-optimizations", "--draft-attn-impl", "eager"])


def test_master_off_during_resume_disables_inherited_members_and_roundtrips(tmp_path):
    config = TrainConfig.resolve(
        [
            *BASE,
            "--fsdp-shard",
            "--activation-checkpointing",
            "--candidate-head",
            "--leagle-safe-optimizations",
            "--leagle-numerical-optimizations",
        ]
    )
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    saved = yaml.safe_load(path.read_text())["train"]["trainer"]
    assert set(NUMERICAL_OPTIONS) | set(SAFE_OPTIONS) <= saved.keys()
    restored = TrainConfig.resolve(["--config", str(path)])
    assert restored.flatten() == config.flatten()
    assert restored.dump_yaml() == config.dump_yaml()

    off = TrainConfig.resolve(
        [
            "--config",
            str(path),
            "--no-leagle-numerical-optimizations",
            "--leagle-chunked-loss",  # The master wins over a member on the CLI too.
        ]
    )
    flat = off.flatten()
    assert all(not flat[name] for name in NUMERICAL_OPTIONS)
    assert flat["leagle_candidate_backward_recompute"]
    assert not flat["leagle_teacher_cache"]
    assert not flat["leagle_cache_attention_masks"]
    assert flat["leagle_share_output_head"]
    assert flat["leagle_fsdp_keep_forward_params"]
    path.write_text(off.dump_yaml())
    assert TrainConfig.resolve(["--config", str(path)]).flatten() == flat
    assert TrainConfig.from_flat(flat).flatten() == flat


def test_yaml_null_releases_master_for_individual_controls(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "train": {
                    "speculator_type": "leagle",
                    "verifier": {"verifier_name_or_path": "dummy"},
                    "trainer": {
                        "leagle_numerical_optimizations": None,
                        "leagle_chunked_loss": True,
                    },
                }
            }
        )
    )
    config = TrainConfig.resolve(["--config", str(path)])
    assert config.trainer.leagle_numerical_optimizations is None
    assert config.trainer.leagle_chunked_loss


def test_old_individual_run_can_be_switched_off_without_inherited_dependencies(
    tmp_path,
):
    config = TrainConfig.resolve(
        [
            *BASE,
            "--candidate-head",
            "--activation-checkpointing",
            "--leagle-chunked-loss",
            "--leagle-teacher-cache",
            "--leagle-sparse-attention",
            "--leagle-cache-attention-masks",
            "--leagle-selective-checkpoint",
            "--leagle-candidate-grad-addmm",
        ]
    )
    path = tmp_path / "old-run.yaml"
    path.write_text(config.dump_yaml())
    updated = TrainConfig.resolve(
        ["--config", str(path), "--no-leagle-numerical-optimizations"]
    ).flatten()
    assert all(not updated[name] for name in NUMERICAL_OPTIONS)
    assert not updated["leagle_teacher_cache"]
    assert not updated["leagle_cache_attention_masks"]
    assert not updated["leagle_selective_checkpoint"]
    assert updated["leagle_candidate_backward_recompute"]


@pytest.mark.parametrize("numerical", [False, True])
def test_checkpoint_candidate_capability_is_resolved_before_trainer_kwargs(numerical):
    config = TrainConfig.resolve(
        [
            *BASE,
            "--loss-implementation",
            "eager",
            "--leagle-numerical-optimizations"
            if numerical
            else "--no-leagle-numerical-optimizations",
        ]
    )
    args = Namespace(**config.flatten())
    assert not args.candidate_head
    model = SimpleNamespace(
        config=SimpleNamespace(
            speculators_model_type="leagle",
            candidate_head=True,
        )
    )
    apply_runtime_optimization_groups(config, args, model)
    assert args.candidate_head
    assert config.draft.candidate_head
    assert args.leagle_candidate_backward_recompute is not numerical
    train, val = LEagleDraftModel.get_trainer_kwargs(**vars(args))
    for call in (train, val):
        assert call["leagle_candidate_backward_recompute"] is not numerical
        assert call["leagle_candidate_grad_addmm"] is numerical
        assert call["leagle_align_candidate_chunks"] is numerical
        assert not any(name in call for name in GROUP_DEFAULTS)
    saved = yaml.safe_load(config.dump_yaml())["train"]["trainer"]
    assert saved["leagle_candidate_backward_recompute"] is not numerical


def test_groups_do_not_modify_architecture_optimizer_or_loss():
    config = TrainConfig.resolve([*BASE, "--num-layers", "3", "--num-loops", "2"])
    grouped = TrainConfig.resolve(
        [
            *BASE,
            "--num-layers",
            "3",
            "--num-loops",
            "2",
            "--leagle-safe-optimizations",
            "--leagle-numerical-optimizations",
        ]
    )
    for group in ("draft", "loss", "optimizer", "scheduler", "data"):
        assert getattr(config, group) == getattr(grouped, group)


@pytest.mark.parametrize("numerical", [False, True])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("amp", [False, True])
def test_grouped_train_backward_and_validation_execute(numerical, candidate, amp):
    torch.manual_seed(41)
    torch.set_num_threads(1)
    model = make_model(layers=1, loops=2, attention="sdpa", candidate_head=candidate)
    model.set_activation_checkpointing(True)
    config = TrainConfig.resolve(
        [
            *BASE,
            "--loss-implementation",
            "eager",
            "--loss-fn",
            '{"ce":0.1,"tv":0.9}',
            "--activation-checkpointing",
            "--leagle-safe-optimizations",
            "--leagle-loss-chunk-size",
            "3",
            "--leagle-attn-chunk-size",
            "2",
            "--hidden-states-dtype",
            "bfloat16" if amp else "float32",
            "--leagle-numerical-optimizations"
            if numerical
            else "--no-leagle-numerical-optimizations",
        ]
    )
    args = Namespace(**config.flatten())
    apply_runtime_optimization_groups(config, args, model)
    if args.leagle_share_output_head:
        reference = model.share_output_head()
        model.validate_shared_output_head(reference)
    train, val = LEagleDraftModel.get_trainer_kwargs(**vars(args))
    inputs = packed_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        _, loss, _ = model(**inputs, **train)
    (loss * 0.37).backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.fc.weight.grad).all()
    if candidate:
        assert torch.isfinite(model.candidate_head.up_proj.weight.grad).all()
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        _, val_loss, _ = model(**inputs, **val)
    assert torch.isfinite(val_loss)
