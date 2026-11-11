"""Sublayer policy persists as architecture through both launch paths and resume."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from speculators import SpeculatorModelConfig
from speculators.train.config import TrainConfig
from speculators.train.legacy_cli import parse_train_args
from speculators.train.model_config import plan_pretrained_config_overrides
from tests.standalone.test_qwen4b_bestarch_scripts import run_shell
from tests.unit.models.test_leagle import make_model
from tests.unit.train.config.test_eagle_train_scripts import launch_args

ROOT = Path(__file__).resolve().parents[3]
BASE = ["--verifier-name-or-path", "fixture", "--speculator-type", "leagle"]


@pytest.mark.parametrize("mode", ["full", "attn", "ffn"])
def test_parsers_checkpoint_config_and_override_policy(mode, tmp_path):
    flags = [*BASE, "--loop-mode", mode, "--num-loops", "3"]
    config = TrainConfig.resolve(flags)
    assert config.draft.loop_mode == parse_train_args(flags).loop_mode == mode
    path = tmp_path / "run.yaml"
    path.write_text(config.dump_yaml())
    assert TrainConfig.resolve(["--config", str(path)]).draft.loop_mode == mode
    model = make_model(layers=2, loops=3, loop_mode=mode)
    model.config.save_pretrained(tmp_path)
    restored = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert restored.loop_mode == mode
    assert json.loads((tmp_path / "config.json").read_text())["loop_mode"] == mode
    args = SimpleNamespace(**TrainConfig.resolve(BASE).flatten())
    inherited, changes = plan_pretrained_config_overrides(args, restored, set())
    assert inherited["loop_mode"] == mode
    assert not changes
    args.loop_mode = "attn" if mode == "ffn" else "ffn"
    with pytest.raises(ValueError, match="loop-mode"):
        plan_pretrained_config_overrides(args, restored, {"loop_mode"})


def test_old_checkpoint_without_mode_retains_full_stack_execution(tmp_path):
    original = make_model(layers=2, loops=3)
    saved = copy.deepcopy(original.config.to_dict())
    saved.pop("loop_mode")
    (tmp_path / "config.json").write_text(json.dumps(saved))
    restored = SpeculatorModelConfig.from_pretrained(tmp_path)
    assert restored.loop_mode == "full"
    assert restored.attention_layer_indices == (0, 1, 0, 1, 0, 1)


@pytest.mark.parametrize(
    "extra",
    [
        ["--loop-mode", "unknown"],
        ["--loop-mode", "attn", "--speculator-type", "eagle3"],
        ["--loop-mode", "ffn", "--speculator-type", "dspark"],
    ],
)
def test_invalid_modes_and_other_algorithms_are_rejected(extra):
    with pytest.raises(SystemExit):
        TrainConfig.resolve([*BASE, *extra])
    with pytest.raises(SystemExit):
        parse_train_args([*BASE, *extra])


@pytest.mark.parametrize("mode", ["full", "attn", "ffn"])
def test_script_modes_use_separate_paths_and_resume_saved_policy(mode, tmp_path):
    config = TrainConfig.resolve(
        launch_args("leagle", MODEL="fixture", NUM_LOOPS="3", LOOP_MODE=mode)
    )
    assert config.draft.loop_mode == mode
    suffix = "" if mode == "full" else f"_{mode}"
    flat = config.flatten()
    assert f"_r3{suffix}_fullvocab" in str(flat["save_path"])
    server = (
        (ROOT / "examples/train/leagle_qwen3_4b_700k/server.sh")
        .read_text()
        .split("# ============ Launch ============")[0]
    )
    result = run_shell(
        f'NUM_LOOPS=3\nLOOP_MODE={mode}\n{server}\nprintf "%s" "$HIDDEN_STATES_PATH"\n'
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(flat["hidden_states_path"])
    checkpoint = tmp_path / "checkpoints"
    checkpoint.mkdir()
    (checkpoint / "run.yaml").write_text(config.dump_yaml())
    resumed = TrainConfig.resolve(
        launch_args(
            "leagle", RESUME="1", OUTPUT_DIR=tmp_path.as_posix(), LOOP_MODE="full"
        )
    )
    assert resumed.draft.loop_mode == mode
    assert resumed.draft.num_loops == 3
    # Candidate-only stage inherits architecture from the saved draft.
    candidate = launch_args(
        "leagle",
        MODEL="fixture",
        CANDIDATE_HEAD_ONLY="1",
        DRAFT_MODEL="draft",
        LOOP_MODE=mode,
    )
    assert "--loop-mode" not in candidate
