"""Exercise the 4B comparisons with Bash, never live training or serving."""

# ruff: noqa: PT009 -- Also runnable with stdlib unittest.

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "examples/train/qwen3_4b_bestarch"
BASH = (
    str(Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe")
    if os.name == "nt"
    else shutil.which("bash")
)
if BASH and not Path(BASH).is_file():
    BASH = None
TRAINERS = (
    "train_dspark.sh",
    "train_mmuse_legacy_optimizer.sh",
    "train_mmuse_upstream_optimizer.sh",
)
# Do not let the host user's training settings leak into the contract tests.
SETTING_KEYS = [
    "MODEL",
    "DATA_PATH",
    "OUTPUT_ROOT",
    "OUTPUT_DIR",
    "VLLM_PORT",
    "VLLM_ENDPOINT",
    "VLLM_NPUS",
    "TRAIN_NPUS",
    "DSPARK_LR",
    "DSPARK_WEIGHT_DECAY",
    "DSPARK_SCHEDULER_TYPE",
    "DSPARK_WARMUP_RATIO",
    "SERVER_READY_TIMEOUT",
    "SERVER_PROVENANCE_DIR",
    "SEQ_LENGTH",
    "EPOCHS",
    "LR",
    "MUON_LR",
    "WEIGHT_DECAY",
    "MUON_WEIGHT_DECAY",
    "SEED",
    "LOGGER",
    "DRAFT_VOCAB_SIZE",
    "DRY_RUN",
    "DUMP_CONFIG",
    "BASH_ENV",
    "ENV",
]


def run_recipe(name, *, settings=None, prelude="", trailer=""):
    """Run real entrypoints with a clean environment and optional fake commands."""
    if not BASH:
        raise unittest.SkipTest("Bash is required for shell wiring tests")
    environment = {
        key: value for key, value in os.environ.items() if key not in SETTING_KEYS
    }
    environment.update(settings or {})
    shell = f"{prelude}\nsource {shlex.quote((RECIPES / name).as_posix())}\n{trailer}\n"
    with tempfile.TemporaryDirectory() as unrelated_cwd:
        return subprocess.run(  # noqa: S603 -- Fixed entrypoints with fake/dry launches.
            [BASH, "--noprofile", "--norc"],
            input=shell,
            cwd=unrelated_cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )


def preview_command(name, **settings):
    result = run_recipe(name, settings={**settings, "DRY_RUN": "1"})
    if result.returncode:
        raise AssertionError(result.stderr)
    return shlex.split(result.stdout.strip())


def training_arguments(name, **settings):
    command = preview_command(name, **settings)
    return command[command.index("-m") + 2 :]


@unittest.skipUnless(BASH, "Bash is required for shell wiring tests")
class Qwen4BBestarchScriptTests(unittest.TestCase):
    def test_all_shell_files_parse(self):
        for path in RECIPES.glob("*.sh"):
            with self.subTest(path=path.name):
                result = subprocess.run(  # noqa: S603 -- Syntax check, no execution.
                    [BASH, "-n", path.as_posix()],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_mmuse_training_budget_and_no_baseline_extensions(self):
        for name in TRAINERS[1:]:
            args = training_arguments(name)
            for key, value in {
                "--training-recipe": "legacy",
                "--loss-implementation": "legacy",
                "--lr": "6e-5",
                "--muon-lr": "6e-4",
                "--total-seq-len": "3072",
                "--epochs": "10",
                "--block-size": "7",
                "--max-anchors": "512",
                "--num-layers": "5",
                "--seed": "42",
            }.items():
                self.assertEqual(args[args.index(key) + 1], value, name)
            self.assertNotIn("--draft-vocab-size", args)
            self.assertIn("--no-resume-from-checkpoint", args)
            self.assertFalse(any("moe" in value for value in args))
        baseline = training_arguments(TRAINERS[0])
        self.assertFalse(
            any(arg.startswith(("--correction-", "--dflash2-")) for arg in baseline)
        )
        self.assertNotIn("--enable-correction-head", baseline)
        self.assertNotIn("--dflash-gated-layer-fusion", baseline)

    def test_all_trainers_use_full_attention_with_sdpa(self):
        for name in TRAINERS:
            with self.subTest(trainer=name):
                args = training_arguments(name)
                layers = int(args[args.index("--num-layers") + 1])
                start = args.index("--full-attention-indices") + 1
                self.assertEqual(args.count("--full-attention-indices"), 1)
                self.assertEqual(
                    args[start : start + layers], [str(i) for i in range(layers)]
                )
                self.assertEqual(args[args.index("--draft-attn-impl") + 1], "sdpa")

    def test_mmuse_commands_only_differ_in_policy_and_output_labels(self):
        old, new = (training_arguments(name) for name in TRAINERS[1:])
        self.assertEqual(len(old), len(new))
        changed = {
            old[index - 1]
            for index, pair in enumerate(zip(old, new, strict=True))
            if pair[0] != pair[1]
        }
        self.assertEqual(
            changed,
            {"--muon-parameter-policy", "--save-path", "--log-dir", "--run-name"},
        )
        self.assertEqual(old[old.index("--muon-parameter-policy") + 1], "legacy")
        self.assertEqual(new[new.index("--muon-parameter-policy") + 1], "upstream")

    def test_server_matches_training_target_and_writes_provenance(self):
        server = preview_command("server.sh")
        train = preview_command(TRAINERS[1])
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=0,1", server)
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7", train)
        self.assertEqual(
            server[server.index("train") + 1],
            train[train.index("--verifier-name-or-path") + 1],
        )
        for command in (server, train):
            start = command.index("--target-layer-ids") + 1
            self.assertEqual(command[start : start + 5], ["1", "9", "17", "25", "33"])
        self.assertEqual(server[server.index("--port") + 1], "8001")
        self.assertEqual(server[server.index("--data-parallel-size") + 1], "2")
        self.assertIn("--provenance-dir", server)
        self.assertNotIn("--spec-model", server)

    def test_overrides_preserve_spaces_and_derive_worker_count(self):
        command = preview_command(
            TRAINERS[1],
            TRAIN_NPUS="10,12",
            MODEL="/weights/Qwen 4B",
            DATA_PATH="/data/prepared samples",
            OUTPUT_DIR="/outputs/new run",
            DRAFT_VOCAB_SIZE="32000",
            VLLM_ENDPOINT="http://target:9000/v1",
        )
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "2")
        self.assertIn("/weights/Qwen 4B", command)
        self.assertIn("/data/prepared samples", command)
        self.assertIn("/outputs/new run/checkpoints", command)
        self.assertIn("http://target:9000/v1", command)
        self.assertEqual(command[command.index("--draft-vocab-size") + 1], "32000")

    def test_invalid_device_list_fails_before_launch(self):
        result = run_recipe(
            TRAINERS[0], settings={"TRAIN_NPUS": "2,,3", "DRY_RUN": "1"}
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("TRAIN_NPUS", result.stderr)

    def test_dspark_changes_only_script_parameters_on_existing_training_entrypoint(
        self,
    ):
        command = preview_command(
            TRAINERS[0], LR="0.1", MUON_LR="0.2", WEIGHT_DECAY="0.3"
        )
        self.assertEqual(command[command.index("-m") + 1], "speculators.train")
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7", command)
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "6")
        for flag, value in {
            "--training-recipe": "legacy",
            "--loss-implementation": "legacy",
            "--optimizer": "muon",
            "--muon-parameter-policy": "legacy",
            "--lr": "6e-4",
            "--muon-lr": "6e-4",
            "--weight-decay": "0.0",
            "--muon-weight-decay": "0.0",
            "--scheduler-type": "cosine",
            "--scheduler-warmup-ratio": "0.04",
            "--total-seq-len": "3072",
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--draft-vocab-size", command)
        self.assertIn("../../datasets/open_perfectblend_qwen3_4b_700k", command)
        start = command.index("--full-attention-indices") + 1
        self.assertEqual(command[start : start + 5], ["0", "1", "2", "3", "4"])
        self.assertTrue(any("dspark_custom/checkpoints" in arg for arg in command))
        twelve = preview_command(
            TRAINERS[0], TRAIN_NPUS=",".join(str(i) for i in range(12))
        )
        self.assertEqual(twelve[twelve.index("--nproc_per_node") + 1], "12")

    def test_dspark_explicit_overrides_and_shared_budget(self):
        command = preview_command(
            TRAINERS[0],
            TRAIN_NPUS="10,11",
            SEQ_LENGTH="2048",
            DSPARK_LR="0.0003",
            DSPARK_WEIGHT_DECAY="0.02",
            DSPARK_SCHEDULER_TYPE="linear",
            DSPARK_WARMUP_RATIO="0.02",
            DATA_PATH="/data/original",
            DRAFT_VOCAB_SIZE="32000",
        )
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=10,11", command)
        for flag, value in {
            "--total-seq-len": "2048",
            "--lr": "0.0003",
            "--muon-lr": "0.0003",
            "--weight-decay": "0.02",
            "--muon-weight-decay": "0.02",
            "--scheduler-type": "linear",
            "--scheduler-warmup-ratio": "0.02",
            "--data-path": "/data/original",
            "--draft-vocab-size": "32000",
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_dspark_overrides_do_not_change_mmuse_commands(self):
        for name in TRAINERS[1:]:
            self.assertEqual(
                preview_command(name),
                preview_command(
                    name,
                    DSPARK_LR="0.03",
                    DSPARK_WEIGHT_DECAY="0.5",
                    DSPARK_SCHEDULER_TYPE="none",
                    DSPARK_WARMUP_RATIO="0.2",
                ),
            )

    def test_server_refuses_an_existing_endpoint(self):
        result = run_recipe(
            "server.sh", prelude="curl() { return 0; }\npython() { return 97; }"
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("already responds", result.stderr)

    def test_server_reports_early_exit_instead_of_waiting_forever(self):
        result = run_recipe(
            "server.sh",
            prelude=r"""
curl() { return 1; }
env() ( while [[ "$1" == *=* ]]; do export "$1"; shift; done; "$@"; )
python() { return 27; }
sleep() { command sleep 0.01; }
""",
        )
        self.assertEqual(result.returncode, 27, result.stderr)
        self.assertIn("exited before becoming ready", result.stderr)

    def test_training_refuses_missing_data_and_existing_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "prepared data"
            data.mkdir()
            output = root / "run"
            settings = {
                "DATA_PATH": str(root / "absent"),
                "OUTPUT_DIR": output.as_posix(),
            }
            missing = run_recipe(TRAINERS[0], settings=settings)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("Arrow data not found", missing.stderr)
            (output / "checkpoints").mkdir(parents=True)
            settings["DATA_PATH"] = data.as_posix()
            existing = run_recipe(TRAINERS[0], settings=settings)
            self.assertEqual(existing.returncode, 2)
            self.assertIn("already used", existing.stderr)


if __name__ == "__main__":
    unittest.main()
