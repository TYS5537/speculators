"""Exercise the actual two launch scripts through their model-free dry run."""

import os
import runpy
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.leagle import LEagleDraftModel
from speculators.train.config import TrainConfig
from tests.standalone.test_qwen4b_bestarch_scripts import training_arguments

ROOT = Path(__file__).resolve().parents[4]
BASH = shutil.which("bash") or "C:/Program Files/Git/bin/bash.exe"


def launch_command(algorithm, **overrides):
    if not Path(BASH).is_file():
        pytest.skip("bash unavailable")
    result = subprocess.run(  # noqa: S603 -- Repository script in non-executing mode.
        [BASH, str(ROOT / f"examples/train/{algorithm}_qwen3_4b_700k/train.sh")],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", **overrides},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout)


def launch_args(algorithm, **overrides):
    command = launch_command(algorithm, **overrides)
    return command[command.index("speculators.train") + 1 :]


def test_candidate_head_train_switch_reaches_model_config_and_resume(tmp_path):
    flags = launch_args(
        "leagle",
        MODEL="model",
        CANDIDATE_HEAD="1",
        CANDIDATE_HEAD_RANK="64",
        CANDIDATE_TOP_K="32",
        CANDIDATE_HEAD_LOSS_WEIGHT="0.5",
        CANDIDATE_HEAD_CHUNK_SIZE="16",
    )
    config = TrainConfig.resolve(flags)
    assert config.draft.candidate_head
    assert config.draft.candidate_head_rank == 64
    assert config.draft.candidate_head_top_k == 32
    assert config.draft.candidate_head_loss_weight == 0.5
    assert config.draft.candidate_head_chunk_size == 16
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(config.dump_yaml())
    resumed = TrainConfig.resolve(
        launch_args(
            "leagle",
            MODEL="model",
            RESUME="1",
            OUTPUT_DIR=tmp_path.as_posix(),
            CANDIDATE_HEAD="0",
        )
    )
    assert resumed.draft.candidate_head
    assert resumed.draft.candidate_head_rank == 64
    assert resumed.draft.candidate_head_loss_weight == 0.5


def test_eagle_script_ignores_leagle_candidate_environment():
    config = TrainConfig.resolve(
        launch_args("eagle3", MODEL="model", CANDIDATE_HEAD="1")
    )
    assert not config.draft.candidate_head


def test_leagle_master_script_switches_and_resume(tmp_path):
    config = TrainConfig.resolve(
        launch_args(
            "leagle",
            MODEL="model",
            CANDIDATE_HEAD="1",
            LEAGLE_SAFE_OPTIMIZATIONS="1",
            LEAGLE_NUMERICAL_OPTIMIZATIONS="1",
        )
    )
    assert config.trainer.leagle_teacher_cache
    assert config.trainer.leagle_sparse_attention
    assert config.trainer.leagle_candidate_grad_addmm
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(config.dump_yaml())
    inherited = TrainConfig.resolve(
        launch_args(
            "leagle",
            RESUME="1",
            OUTPUT_DIR=tmp_path.as_posix(),
        )
    )
    for name, value in config.trainer.model_dump().items():
        if name.startswith("leagle_"):
            assert getattr(inherited.trainer, name) == value
    assert not inherited.trainer.no_resume_from_checkpoint
    off = TrainConfig.resolve(
        launch_args(
            "leagle",
            RESUME="1",
            OUTPUT_DIR=tmp_path.as_posix(),
            LEAGLE_NUMERICAL_OPTIMIZATIONS="0",
            LEAGLE_SPARSE_ATTENTION="1",
        )
    )
    assert off.trainer.leagle_safe_optimizations
    assert not off.trainer.leagle_sparse_attention
    assert not off.trainer.leagle_teacher_cache
    assert off.trainer.leagle_candidate_backward_recompute
    assert off.trainer.leagle_share_output_head


def test_eagle_script_ignores_leagle_optimization_masters():
    config = TrainConfig.resolve(
        launch_args(
            "eagle3",
            MODEL="model",
            LEAGLE_SAFE_OPTIMIZATIONS="1",
            LEAGLE_NUMERICAL_OPTIMIZATIONS="1",
        )
    )
    assert config.trainer.leagle_safe_optimizations is None
    assert config.trainer.leagle_numerical_optimizations is None


def test_leagle_optimization_script_resume_inherits_or_explicitly_overrides(tmp_path):
    names = (
        "LEAGLE_REUSE_TRAINING_KV",
        "LEAGLE_CHUNKED_LOSS",
        "LEAGLE_SPARSE_ATTENTION",
        "LEAGLE_CANDIDATE_GRAD_ADDMM",
        "LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE",
        "LEAGLE_TEACHER_CACHE",
        "LEAGLE_MASKED_HEAD",
        "LEAGLE_ALIGN_CANDIDATE_CHUNKS",
        "LEAGLE_SELECTIVE_CHECKPOINT",
        "LEAGLE_CACHE_ATTENTION_MASKS",
    )
    defaults = TrainConfig.resolve(launch_args("leagle", MODEL="model")).flatten()
    assert all(defaults[name.lower()] is False for name in names)
    original = TrainConfig.resolve(
        launch_args(
            "leagle",
            MODEL="model",
            **dict.fromkeys(names, "1"),
            CANDIDATE_HEAD="1",
            LEAGLE_LOSS_CHUNK_SIZE="256",
            LEAGLE_ATTN_CHUNK_SIZE="64",
            LEAGLE_TEACHER_CACHE_MB="512",
            LEAGLE_ATTENTION_MASK_CACHE_MB="7",
        )
    )
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(original.dump_yaml())
    resumed = TrainConfig.resolve(
        launch_args("leagle", RESUME="1", OUTPUT_DIR=tmp_path.as_posix())
    ).flatten()
    assert all(resumed[name.lower()] is True for name in names)
    assert resumed["leagle_loss_chunk_size"] == 256
    assert resumed["leagle_teacher_cache_mb"] == 512
    assert resumed["leagle_attention_mask_cache_mb"] == 7
    overridden = TrainConfig.resolve(
        launch_args(
            "leagle",
            RESUME="1",
            OUTPUT_DIR=tmp_path.as_posix(),
            LEAGLE_CHUNKED_LOSS="0",
            LEAGLE_CANDIDATE_BACKWARD_RECOMPUTE="0",
            LEAGLE_ATTN_CHUNK_SIZE="32",
            LEAGLE_MASKED_HEAD="0",
            LEAGLE_TEACHER_CACHE_MB="256",
            LEAGLE_ALIGN_CANDIDATE_CHUNKS="0",
            LEAGLE_SELECTIVE_CHECKPOINT="0",
            LEAGLE_CACHE_ATTENTION_MASKS="0",
            LEAGLE_ATTENTION_MASK_CACHE_MB="3",
        )
    ).flatten()
    assert not overridden["leagle_chunked_loss"]
    assert not overridden["leagle_candidate_backward_recompute"]
    assert overridden["leagle_reuse_training_kv"]
    assert overridden["leagle_sparse_attention"]
    assert overridden["leagle_attn_chunk_size"] == 32
    assert not overridden["leagle_masked_head"]
    assert overridden["leagle_teacher_cache_mb"] == 256
    assert overridden["leagle_attention_mask_cache_mb"] == 3
    assert not overridden["leagle_align_candidate_chunks"]
    assert not overridden["leagle_selective_checkpoint"]
    assert not overridden["leagle_cache_attention_masks"]


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_default_recipes_resolve_without_changing_training_budget(algorithm):
    flags = launch_args(algorithm, MODEL="model")
    args = TrainConfig.resolve(flags).flatten()
    assert args["optimizer"] == "adamw"
    assert args["lr"] == 6e-4
    assert not any(flag.startswith("--muon-") for flag in flags)
    assert args["weight_decay"] == 0.0
    assert args["adam_beta1"] == 0.9
    assert args["epochs"] == 5
    assert args["ttt_steps"] == 7
    assert args["log_freq"] == (10 if algorithm == "leagle" else 1)
    assert args["num_layers"] == args["num_loops"] == 1
    assert args["draft_vocab_size"] == 151936
    assert args["target_layer_ids"] == (
        [2, 18, 33] if algorithm == "eagle3" else [1, 9, 17, 25, 33]
    )
    assert args["full_attention_indices"] == [0]
    assert args["training_recipe"] == "upstream"  # Retain size-keyed vocab caches.
    assert args["fsdp_shard"] is (algorithm == "leagle")
    assert args["activation_checkpointing"] is (algorithm == "leagle")
    if algorithm == "eagle3":
        assert args["adam_beta2"] == 0.95
        assert args["max_grad_norm"] == 0.5
        assert args["loss_fn"] == "kl_div"
        assert args["loss_implementation"] == "eager"
        assert args["scheduler_type"] == "linear"
        assert args["scheduler_warmup_ratio"] == 0.015
    else:
        dspark = TrainConfig.resolve(training_arguments("train_dspark.sh")).flatten()
        for key in (
            "optimizer",
            "lr",
            "weight_decay",
            "adam_beta1",
            "adam_beta2",
            "max_grad_norm",
            "loss_fn",
            "loss_implementation",
            "scheduler_type",
            "scheduler_warmup_ratio",
            "scheduler_total_steps",
            "scheduler_warmup_steps",
            "scheduler_num_cosine_cycles",
            "noise_std",
            "train_data_ratio",
            "total_seq_len",
        ):
            assert args[key] == dspark[key], key
        # Explicit legacy CE/TV must reach the model despite upstream data handling.
        leagle_train, _ = LEagleDraftModel.get_trainer_kwargs(**args)
        dspark_train, _ = DSparkDraftModel.get_trainer_kwargs(**dspark)
        assert leagle_train["loss_config"] == dspark_train["loss_config"]


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_script_memory_retention_and_loss_overrides(algorithm):
    flags = launch_args(
        algorithm,
        MODEL="model",
        NUM_LAYERS="2",
        NUM_LOOPS="3",
        LR="0.00003",
        ADAM_BETA1="0.85",
        ADAM_BETA2="0.98",
        MAX_GRAD_NORM="0.3",
        DRAFT_VOCAB_SIZE="32000",
        LOSS_FN="tv",
        FSDP_SHARD="1",
        ACTIVATION_CHECKPOINTING="1",
    )
    args = TrainConfig.resolve(flags).flatten()
    assert args["fsdp_shard"]
    assert args["activation_checkpointing"]
    assert not args["save_best"]
    assert args["checkpoint_freq"] == 1
    assert args["epochs"] == 5
    assert args["lr"] == 0.00003
    assert args["adam_beta1"] == 0.85
    assert args["adam_beta2"] == 0.98
    assert args["max_grad_norm"] == 0.3
    assert args["draft_vocab_size"] == 32000
    assert args["loss_fn"] == "tv"
    assert args["num_layers"] == 2
    assert args["num_loops"] == (3 if algorithm == "leagle" else 1)
    assert args["draft_attn_impl"] == "sdpa"


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_resume_uses_saved_recipe_and_can_disable_best_only(algorithm, tmp_path):
    original = TrainConfig.resolve(
        launch_args(
            algorithm,
            MODEL="model",
            NUM_LAYERS="2",
            NUM_LOOPS="3",
            LR="0.00003",
            ADAM_BETA1="0.85",
            ADAM_BETA2="0.98",
            MAX_GRAD_NORM="0.3",
            DRAFT_VOCAB_SIZE="32000",
        )
    )
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(original.dump_yaml())
    flags = launch_args(algorithm, RESUME="1", OUTPUT_DIR=tmp_path.as_posix())
    args = TrainConfig.resolve(flags).flatten()
    assert not args["no_resume_from_checkpoint"]
    assert not args["save_best"]
    assert args["lr"] == 0.00003  # Original recipe, not the script's default.
    assert args["adam_beta1"] == 0.85
    assert args["adam_beta2"] == 0.98
    assert args["max_grad_norm"] == 0.3
    assert args["draft_vocab_size"] == 32000  # Resuming does not resize the head.
    assert args["num_layers"] == 2
    assert args["num_loops"] == (3 if algorithm == "leagle" else 1)


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_resume_old_muon_yaml_keeps_historical_optimizer_and_clipping(
    algorithm, tmp_path
):
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(
        f"train:\n  speculator_type: {algorithm}\n  training_recipe: upstream\n"
        "  verifier:\n    verifier_name_or_path: model\n"
        "  optimizer:\n    optimizer: muon\n    lr: 0.0001\n    muon_lr: 0.0001\n"
        "  loss:\n    loss_fn: kl_div\n    loss_implementation: eager\n"
    )
    args = TrainConfig.resolve(
        launch_args(algorithm, RESUME="1", OUTPUT_DIR=tmp_path.as_posix())
    ).flatten()
    assert args["optimizer"] == "muon"
    assert args["lr"] == args["muon_lr"] == 1e-4
    assert args["adam_beta1"] == 0.9
    assert args["adam_beta2"] == 0.999
    assert args["max_grad_norm"] == 1.0
    assert args["loss_fn"] == "kl_div"
    assert args["loss_implementation"] == "eager"


def test_memory_flags_can_be_disabled():
    args = TrainConfig.resolve(
        launch_args(
            "leagle", MODEL="model", FSDP_SHARD="0", ACTIVATION_CHECKPOINTING="0"
        )
    ).flatten()
    assert not args["fsdp_shard"]
    assert not args["activation_checkpointing"]


def test_priority_switches_keep_delete_policy_and_resume_overrides(tmp_path):
    flags = (
        "LEAGLE_TEACHER_CACHE",
        "LEAGLE_TEACHER_CACHE_SUPERVISED",
        "LEAGLE_SHARED_LOSS_NORMALIZATION",
        "LEAGLE_STREAM_MAIN_LOSS",
        "LEAGLE_SHARE_OUTPUT_HEAD",
        "LEAGLE_FSDP_KEEP_FORWARD_PARAMS",
    )
    config = TrainConfig.resolve(
        launch_args(
            "leagle", MODEL="model", FSDP_SHARD="1", **dict.fromkeys(flags, "1")
        )
    )
    assert all(config.flatten()[name.lower()] for name in flags)
    assert config.flatten()["on_generate"] == "delete"
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(config.dump_yaml())
    for overrides, enabled in (({}, True), (dict.fromkeys(flags, "0"), False)):
        resumed = TrainConfig.resolve(
            launch_args(
                "leagle", RESUME="1", OUTPUT_DIR=tmp_path.as_posix(), **overrides
            )
        ).flatten()
        assert all(resumed[name.lower()] is enabled for name in flags)
        assert resumed["on_generate"] == "delete"


def test_leagle_log_frequency_can_be_overridden_on_new_and_resumed_runs(tmp_path):
    original = TrainConfig.resolve(launch_args("leagle", MODEL="model", LOG_FREQ="3"))
    assert original.flatten()["log_freq"] == 3
    path = tmp_path / "checkpoints"
    path.mkdir()
    (path / "run.yaml").write_text(original.dump_yaml())
    resumed = TrainConfig.resolve(
        launch_args("leagle", RESUME="1", OUTPUT_DIR=tmp_path.as_posix(), LOG_FREQ="1")
    )
    assert resumed.flatten()["log_freq"] == 1


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_script_uses_checkout_with_selected_python(algorithm):
    command = launch_command(
        algorithm, PYTHON="/selected env/bin/python", PYTHONPATH="/old-installed-code"
    )
    python_index = command.index("/selected env/bin/python")
    assert command[python_index + 1 : python_index + 3] == [
        "-m",
        "torch.distributed.run",
    ]
    path = next(
        arg.removeprefix("PYTHONPATH=")
        for arg in command[:python_index]
        if arg.startswith("PYTHONPATH=")
    )
    # Bash emits POSIX paths on Windows too. The repo must precede inherited code.
    repo_path = subprocess.check_output(  # noqa: S603 -- Only resolves the test cwd.
        [BASH, "-c", "pwd"], cwd=ROOT, text=True
    ).strip()
    assert path.split(":")[:3] == [
        f"{repo_path}/src",
        f"{repo_path}/hs_connectors/src",
        repo_path,
    ]
    assert path.endswith(":/old-installed-code")


@pytest.mark.parametrize("algorithm", ["eagle3", "leagle"])
def test_actual_module_entrypoint_accepts_script_flags(
    algorithm, monkeypatch, capsys, tmp_path
):
    flags = launch_args(algorithm, MODEL="model")
    monkeypatch.setattr("sys.argv", ["speculators.train", *flags, "--dump-config"])
    with pytest.raises(SystemExit) as result:
        runpy.run_module("speculators.train", run_name="__main__")
    output = capsys.readouterr()
    assert result.value.code == 0, output.err
    path = tmp_path / "run.yaml"
    path.write_text(output.out)
    config = TrainConfig.resolve(["--config", str(path)]).flatten()
    assert config["draft_vocab_size"] == 151936
    assert config["adam_beta1"] == 0.9
    assert config["adam_beta2"] == (0.95 if algorithm == "eagle3" else 0.999)
    assert config["max_grad_norm"] == (0.5 if algorithm == "eagle3" else 1.0)
