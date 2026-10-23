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
    assert args["num_layers"] == args["num_loops"] == 1
    assert args["draft_vocab_size"] == 151936
    assert args["target_layer_ids"] == [2, 18, 33]
    assert args["full_attention_indices"] == [0]
    assert args["training_recipe"] == "upstream"  # Retain size-keyed vocab caches.
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
