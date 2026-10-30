"""Experimental flags stay off unless selected and survive YAML/resume."""

import inspect

import pytest

from speculators.models.leagle import LEagleDraftModel
from speculators.train.config import TrainConfig
from speculators.train.legacy_cli import parse_train_args

BASE = [
    "--verifier-name-or-path",
    "dummy",
    "--speculator-type",
    "leagle",
    "--draft-attn-impl",
    "sdpa",
]
NAMES = (
    "leagle_reuse_training_kv",
    "leagle_chunked_loss",
    "leagle_sparse_attention",
    "leagle_candidate_grad_addmm",
    "leagle_candidate_backward_recompute",
    "leagle_teacher_cache",
    "leagle_masked_head",
    "leagle_teacher_cache_supervised",
    "leagle_shared_loss_normalization",
    "leagle_stream_main_loss",
    "leagle_align_candidate_chunks",
    "leagle_selective_checkpoint",
    "leagle_cache_attention_masks",
)


def test_optimization_defaults_are_off_and_flags_roundtrip(tmp_path):
    defaults = TrainConfig.resolve(BASE).flatten()
    assert all(defaults[name] is False for name in NAMES)
    expected = {
        **dict.fromkeys(NAMES, False),
        "leagle_loss_chunk_size": 128,
        "leagle_attn_chunk_size": 128,
        "leagle_teacher_cache_mb": 1024,
        "leagle_attention_mask_cache_mb": 64,
    }
    call_kwargs, _ = LEagleDraftModel.get_trainer_kwargs(loss_fn="kl_div")
    signature = inspect.signature(LEagleDraftModel.forward)
    for actual in (defaults, vars(parse_train_args(BASE)), call_kwargs):
        assert {name: actual[name] for name in expected} == expected
    assert {name: signature.parameters[name].default for name in expected} == expected
    flags = [
        "--activation-checkpointing",
        *("--" + name.replace("_", "-") for name in NAMES),
    ]
    config = TrainConfig.resolve(
        [
            *BASE,
            *flags,
            "--leagle-loss-chunk-size",
            "32",
            "--leagle-attn-chunk-size",
            "16",
        ]
    )
    legacy = parse_train_args(
        [
            *BASE,
            *flags,
            "--leagle-loss-chunk-size",
            "32",
            "--leagle-attn-chunk-size",
            "16",
        ]
    )
    assert all(getattr(legacy, name) is True for name in NAMES)
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    resumed = TrainConfig.resolve(["--config", str(path)])
    assert resumed.trainer == config.trainer
    train, val = LEagleDraftModel.get_trainer_kwargs(**resumed.flatten())
    assert all(train[name] is True and val[name] is True for name in NAMES)
    assert train["leagle_loss_chunk_size"] == 32
    assert train["leagle_attn_chunk_size"] == 16
    assert train["return_predictions"] is val["return_predictions"] is False
    disabled = TrainConfig.resolve(["--config", str(path), "--no-leagle-chunked-loss"])
    assert not disabled.trainer.leagle_chunked_loss


@pytest.mark.parametrize(
    "extra",
    [
        ["--leagle-reuse-training-kv", "--speculator-type", "eagle3"],
        ["--leagle-sparse-attention", "--draft-attn-impl", "eager"],
        ["--leagle-loss-chunk-size", "0"],
        ["--leagle-attn-chunk-size", "-1"],
        ["--leagle-teacher-cache-mb", "0"],
        ["--leagle-teacher-cache", "--speculator-type", "eagle3"],
        ["--leagle-masked-head", "--speculator-type", "eagle3"],
        ["--leagle-candidate-grad-addmm", "--speculator-type", "eagle3"],
        ["--leagle-candidate-backward-recompute", "--speculator-type", "eagle3"],
        ["--leagle-teacher-cache-supervised"],
        ["--leagle-fsdp-keep-forward-params", "--no-fsdp-shard"],
        ["--leagle-share-output-head", "--speculator-type", "eagle3"],
        ["--leagle-shared-loss-normalization", "--speculator-type", "eagle3"],
        ["--leagle-stream-main-loss", "--speculator-type", "eagle3"],
        ["--leagle-align-candidate-chunks", "--speculator-type", "eagle3"],
        ["--leagle-selective-checkpoint"],
        ["--leagle-selective-checkpoint", "--leagle-sparse-attention"],
        ["--leagle-cache-attention-masks"],
        ["--leagle-attention-mask-cache-mb", "0"],
    ],
)
def test_unsupported_options_fail_in_both_parsers(extra):
    with pytest.raises(SystemExit):
        TrainConfig.resolve([*BASE, *extra])
    with pytest.raises(SystemExit):
        parse_train_args([*BASE, *extra])


def test_runtime_options_are_separate_from_forward_and_survive_resume(tmp_path):
    names = ("leagle_share_output_head", "leagle_fsdp_keep_forward_params")
    flags = ["--" + name.replace("_", "-") for name in names]
    for parse in (
        lambda argv: TrainConfig.resolve(argv).flatten(),
        lambda argv: vars(parse_train_args(argv)),
    ):
        assert all(parse(BASE)[name] is False for name in names)
        assert all(
            parse([*BASE, "--fsdp-shard", *flags])[name] is True for name in names
        )
    config = TrainConfig.resolve([*BASE, "--fsdp-shard", *flags])
    path = tmp_path / "runtime.yaml"
    path.write_text(config.dump_yaml())
    resumed = TrainConfig.resolve(["--config", str(path)])
    assert resumed.trainer == config.trainer
    train, val = LEagleDraftModel.get_trainer_kwargs(**resumed.flatten())
    assert all(name not in train and name not in val for name in names)
    off = TrainConfig.resolve(
        [
            "--config",
            str(path),
            "--no-leagle-share-output-head",
            "--no-leagle-fsdp-keep-forward-params",
        ]
    )
    assert all(off.flatten()[name] is False for name in names)
