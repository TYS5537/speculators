"""Exercise the 4B comparisons with Bash, never live training or serving."""

# ruff: noqa: PT009 -- Also runnable with stdlib unittest.

import os
import re
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
LAUNCH_MARKER = "# ============ Launch ============"


def recipe_source(name, settings=None):
    """Simulate editing scalar settings at the top, without modifying repo files."""
    configuration, launch = (
        (RECIPES / name).read_text(encoding="utf-8").split(LAUNCH_MARKER, 1)
    )
    for key, value in (settings or {}).items():
        configuration, count = re.subn(
            rf"^{re.escape(key)}=.*$",
            lambda match, key=key, value=value: f"{key}={shlex.quote(value)}",
            configuration,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise AssertionError(f"Expected one editable {key} in {name}")
    return configuration + LAUNCH_MARKER + launch


def run_shell(shell):
    """Bash execution is isolated in a temporary cwd, with no implicit startup file."""
    if not BASH:
        raise unittest.SkipTest("Bash is required for shell wiring tests")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}
    }
    with tempfile.TemporaryDirectory() as unrelated_cwd:
        return subprocess.run(  # noqa: S603 -- Fixed recipes with inert launches.
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


def run_recipe(name, *, settings=None, prelude="", trailer=""):
    """Exercise guards/logging with fake commands; never start training/serving."""
    guards = """
curl() { return 1; }
env() { echo 'Unexpected server launch' >&2; return 97; }
nohup() { echo 'Unexpected training launch' >&2; return 97; }
"""
    return run_shell(
        f"{guards}\n{prelude}\n{recipe_source(name, settings)}\n{trailer}\n"
    )


def preview_command(name, **settings):
    # Evaluate the real configuration and direct command with a capture function.
    # Do not run the launch section (filesystem/network/process side effects).
    configuration, launch = recipe_source(name, settings).split(LAUNCH_MARKER, 1)
    pattern = (
        r'^nohup (env .*?) \\\n    > "\$LOG_FILE" 2>&1 &$'
        if name in TRAINERS
        else r"^(env ASCEND_RT_VISIBLE_DEVICES=.*?) &\nVLLM_PID="
    )
    match = re.search(pattern, launch, flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"Expected a direct launch command in {name}")
    result = run_shell(
        configuration
        + "\ncapture() { printf '%s\\0' \"$@\"; }\n"
        + f"capture {match.group(1)}\n"
    )
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.rstrip("\0").split("\0")


def training_arguments(name, **settings):
    command = preview_command(name, **settings)
    return command[command.index("-m") + 2 :]


@unittest.skipUnless(BASH, "Bash is required for shell wiring tests")
class Qwen4BBestarchScriptTests(unittest.TestCase):
    def test_four_standalone_scripts_without_shared_configuration(self):
        self.assertEqual(
            {path.name for path in RECIPES.glob("*.sh")},
            {*TRAINERS, "server.sh"},
        )
        for name in (*TRAINERS, "server.sh"):
            source = recipe_source(name)
            self.assertNotRegex(source, r"(?m)^\s*(?:source|\.)\s")
            self.assertNotIn("run_qwen4b_training", source)
            self.assertNotIn("DRY_RUN", source)
            self.assertNotIn("DUMP_CONFIG", source)
            self.assertIn('MODEL="../../Qwen3-4B"', source)

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
                "--optimizer": "muon",
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

    def test_editable_settings_preserve_spaces_and_explicit_worker_count(self):
        command = preview_command(
            TRAINERS[1],
            TRAIN_NPUS="10,12",
            NUM_TRAIN_NPUS="2",
            MODEL="/weights/Qwen 4B",
            DATA_PATH="/data/prepared samples",
            OUTPUT_DIR="/outputs/new run",
            VLLM_ENDPOINT="http://target:9000/v1",
        )
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "2")
        self.assertIn("/weights/Qwen 4B", command)
        self.assertIn("/data/prepared samples", command)
        self.assertIn("/outputs/new run/checkpoints", command)
        self.assertIn("http://target:9000/v1", command)

    def test_invalid_device_list_or_worker_count_fails_before_launch(self):
        for name in TRAINERS:
            for devices in ("2,,3", "2,3"):
                result = run_recipe(name, settings={"TRAIN_NPUS": devices})
                self.assertEqual(result.returncode, 2)
                self.assertIn("TRAIN_NPUS", result.stderr)

    def test_dspark_changes_only_script_parameters_on_existing_training_entrypoint(
        self,
    ):
        command = preview_command(TRAINERS[0])
        self.assertEqual(command[command.index("-m") + 1], "speculators.train")
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7", command)
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "6")
        for flag, value in {
            "--training-recipe": "legacy",
            "--loss-implementation": "legacy",
            "--optimizer": "adamw",
            "--lr": "6e-4",
            "--weight-decay": "0.0",
            "--scheduler-type": "cosine",
            "--scheduler-warmup-ratio": "0.04",
            "--total-seq-len": "3072",
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--draft-vocab-size", command)
        self.assertFalse(any(arg.startswith("--muon-") for arg in command))
        self.assertIn("../../datasets/open_perfectblend_qwen3_4b_700k", command)
        start = command.index("--full-attention-indices") + 1
        self.assertEqual(command[start : start + 5], ["0", "1", "2", "3", "4"])
        self.assertTrue(any("dspark_custom/checkpoints" in arg for arg in command))
        twelve = preview_command(
            TRAINERS[0],
            TRAIN_NPUS=",".join(str(i) for i in range(12)),
            NUM_TRAIN_NPUS="12",
        )
        self.assertEqual(twelve[twelve.index("--nproc_per_node") + 1], "12")

    def test_dspark_explicit_overrides_and_shared_budget(self):
        command = preview_command(
            TRAINERS[0],
            TRAIN_NPUS="10,11",
            NUM_TRAIN_NPUS="2",
            SEQ_LENGTH="2048",
            LR="0.0003",
            WEIGHT_DECAY="0.02",
            SCHEDULER_TYPE="linear",
            WARMUP_RATIO="0.02",
            DATA_PATH="/data/original",
        )
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=10,11", command)
        for flag, value in {
            "--total-seq-len": "2048",
            "--lr": "0.0003",
            "--weight-decay": "0.02",
            "--scheduler-type": "linear",
            "--scheduler-warmup-ratio": "0.02",
            "--data-path": "/data/original",
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)

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
env() { return 27; }
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
            for name in TRAINERS:
                missing = run_recipe(name, settings=settings)
                self.assertEqual(missing.returncode, 2)
                self.assertIn("Arrow data not found", missing.stderr)
            (output / "checkpoints").mkdir(parents=True)
            settings["DATA_PATH"] = data.as_posix()
            for name in TRAINERS:
                existing = run_recipe(name, settings=settings)
                self.assertEqual(existing.returncode, 2)
                self.assertIn("already used", existing.stderr)

    def test_training_launch_writes_log_pid_and_uses_edited_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "prepared data"
            data.mkdir()
            for name in TRAINERS:
                output = root / name
                settings = {
                    "DATA_PATH": data.as_posix(),
                    "OUTPUT_DIR": output.as_posix(),
                }
                unavailable = run_recipe(name, settings=settings)
                self.assertEqual(unavailable.returncode, 2)
                self.assertIn("Target unavailable", unavailable.stderr)
                self.assertFalse(output.exists())
                launched = run_recipe(
                    name,
                    settings=settings,
                    prelude="""
curl() { return 0; }
nohup() { printf '%s\\0' "$@"; }
""",
                    trailer="wait",
                )
                self.assertEqual(launched.returncode, 0, launched.stderr)
                self.assertTrue(
                    (output / "logs/train.pid").read_text().strip().isdigit()
                )
                logs = list((output / "logs").glob("train_*.log"))
                self.assertEqual(len(logs), 1)
                captured = logs[0].read_text().rstrip("\0").split("\0")
                self.assertEqual(captured, preview_command(name, **settings))


if __name__ == "__main__":
    unittest.main()
