"""Real second-stage launch script in model-free DRY_RUN mode."""

import os
import shlex
import subprocess

import pytest

from speculators.train.config import TrainConfig
from tests.unit.train.config.test_eagle_train_scripts import BASH, ROOT, launch_args


def launch(*, cwd=ROOT, **settings):
    return subprocess.run(  # noqa: S603 -- Script prints argv and exits.
        [
            BASH,
            str(ROOT / "examples/train/leagle_qwen3_4b_700k/train_candidate_head.sh"),
        ],
        cwd=cwd,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "MODEL": "target",
            "DRAFT_MODEL": "draft",
            **settings,
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_standalone_defaults_match_e2e_paths_and_training_budget(tmp_path):
    common = {"MODEL": "../../Qwen3-4B", "DATA_PATH": "../../datasets/prepared_700k"}
    e2e = TrainConfig.resolve(launch_args("leagle", **common)).flatten()
    result = launch(cwd=tmp_path, DRAFT_MODEL="draft with spaces", **common)
    assert result.returncode == 0, result.stderr
    cmd = shlex.split(result.stdout)
    actual = TrainConfig.resolve(cmd[cmd.index("speculators.train") + 1 :]).flatten()
    for name in (
        "verifier_name_or_path",
        "data_path",
        "vllm_endpoint",
        "hidden_states_backend",
        "total_seq_len",
        "epochs",
        "train_data_ratio",
        "ttt_steps",
        "optimizer",
        "lr",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "max_grad_norm",
        "scheduler_type",
        "scheduler_warmup_ratio",
        "draft_attn_impl",
        "fsdp_shard",
    ):
        assert actual[name] == e2e[name], name
    assert actual["from_pretrained"] == "draft with spaces"
    assert actual["candidate_head_only"]
    assert actual["candidate_rerank"]
    assert actual["candidate_head_rank"] == 640
    assert actual["candidate_head_top_k"] == 512
    assert actual["loss_fn"] == "kl_div"
    assert actual["save_path"] == "./output/leagle_candidate_r640/checkpoints"
    assert (
        actual["hidden_states_path"] == "./output/leagle_candidate_r640/hidden_states"
    )
    assert actual["log_dir"] == "./output/leagle_candidate_r640/tensorboard"
    assert "ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7" in cmd


def test_ddp_and_optimization_controls_reach_second_stage():
    result = launch(
        FSDP_SHARD="0",
        LEAGLE_SAFE_OPTIMIZATIONS="1",
        LEAGLE_NUMERICAL_OPTIMIZATIONS="1",
        LEAGLE_DDP_BUCKET_CAP_MB="64",
        CANDIDATE_HEAD_CHUNK_SIZE="512",
    )
    assert result.returncode == 0, result.stderr
    cmd = shlex.split(result.stdout)
    cfg = TrainConfig.resolve(cmd[cmd.index("speculators.train") + 1 :])
    assert not cfg.trainer.fsdp_shard
    assert cfg.trainer.leagle_ddp_gradient_as_bucket_view
    assert cfg.trainer.leagle_ddp_batched_grad_copy
    assert cfg.trainer.leagle_ddp_bucket_cap_mb == 64
    assert cfg.draft.candidate_head_only
    assert cfg.draft.candidate_head_chunk_size == 512
    assert not cfg.trainer.leagle_stream_main_loss
    assert not cfg.trainer.leagle_teacher_cache


@pytest.mark.parametrize("initialization", ["svd", "random"])
def test_source_head_settings_and_main_stage_defaults(initialization):
    result = launch(
        CANDIDATE_HEAD_INIT=initialization,
        CANDIDATE_HEAD_RANK="192",
        CANDIDATE_TOP_K="64",
        CANDIDATE_HEAD_TRAIN_STRIDE="8",
        LR="0.0002",
    )
    assert result.returncode == 0, result.stderr
    cmd = shlex.split(result.stdout)
    argv = cmd[cmd.index("speculators.train") + 1 :]
    cfg = TrainConfig.resolve(argv)
    assert cfg.draft.candidate_head_only
    assert cfg.draft.from_pretrained == "draft"
    assert cfg.draft.candidate_head_init == initialization
    assert cfg.draft.candidate_head_rank == 192
    assert cfg.draft.candidate_head_top_k == 64
    assert cfg.draft.candidate_head_train_stride == 8
    assert cfg.optimizer.lr == 0.0002
    assert not cfg.trainer.activation_checkpointing
    assert cfg.data.noise_std == 0
    for flag in (
        "--num-layers",
        "--num-loops",
        "--draft-arch",
        "--draft-vocab-size",
        "--token-freq-path",
        "--target-layer-ids",
        "--full-attention-indices",
    ):
        assert flag not in argv


def test_resume_inherits_source_policy_and_optimizer(tmp_path):
    result = launch(CANDIDATE_HEAD_INIT="random", CANDIDATE_HEAD_RANK="64", LR="0.0002")
    assert result.returncode == 0, result.stderr
    cmd = shlex.split(result.stdout)
    cfg = TrainConfig.resolve(cmd[cmd.index("speculators.train") + 1 :])
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "run.yaml").write_text(cfg.dump_yaml())
    resumed = launch(
        RESUME="1",
        OUTPUT_DIR=tmp_path.as_posix(),
        DRAFT_MODEL="",
        CANDIDATE_HEAD_INIT="svd",
        LR="0.8",
    )
    assert resumed.returncode == 0, resumed.stderr
    cmd = shlex.split(resumed.stdout)
    loaded = TrainConfig.resolve(cmd[cmd.index("speculators.train") + 1 :])
    assert loaded.draft.candidate_head_only
    assert loaded.draft.from_pretrained == "draft"
    assert loaded.draft.candidate_head_init == "random"
    assert loaded.draft.candidate_head_rank == 64
    assert loaded.optimizer.lr == 0.0002


@pytest.mark.parametrize(
    "settings", [{"DRAFT_MODEL": ""}, {"CANDIDATE_HEAD_INIT": "invalid"}]
)
def test_invalid_stage_two_settings_stop_before_launch(settings):
    result = launch(**settings)
    assert result.returncode == 2
    assert "speculators.train" not in result.stdout
