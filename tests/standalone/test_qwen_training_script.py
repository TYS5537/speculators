"""Exercise Qwen launch arguments without starting training or creating run outputs."""

# ruff: noqa: PT009 -- Also runnable without pytest/torch.

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/train/dspark_qwen3_8b_trainer.sh"
COMMON_ENV = SCRIPT.parent / "common/ascend_training_env.sh"
ASCEND_ENVIRONMENT = {
    "OMP_PROC_BIND": "false",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VE_OMP_NUM_THREADS": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "TASK_QUEUE_ENABLE": "2",
    "ACLNN_CACHE_LIMIT": "100000",
    "NPU_ASD_ENABLE": "0",
    "ASCEND_LAUNCH_BLOCKING": "0",
}
if os.name == "nt":
    BASH = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
    BASH = str(BASH) if BASH.is_file() else None
else:
    BASH = shutil.which("bash")

STUBS = r"""
fixture_initial_pwd="$PWD"
# Redirect log/PID writes into the fixture, without creating the recipe's output.
mkdir() { LOG_DIR="$FIXTURE_ROOT"; }
date() { printf 'fixture\n'; }
nohup() {
  [[ "$PWD" == "$fixture_initial_pwd" ]] || exit 98
  printf '%s\n' "$@" > "$CAPTURE"
  env | while IFS='=' read -r name value; do
    case "$name" in
      OMP_PROC_BIND|OMP_NUM_THREADS|MKL_NUM_THREADS|VE_OMP_NUM_THREADS|\
      PYTORCH_NPU_ALLOC_CONF|TASK_QUEUE_ENABLE|ACLNN_CACHE_LIMIT|\
      NPU_ASD_ENABLE|ASCEND_LAUNCH_BLOCKING)
        printf '%s=%s\n' "$name" "$value" ;;
    esac
  done > "$ENV_CAPTURE"
}
"""


def _shell_environment(overrides=None):
    environment = {**os.environ, **(overrides or {})}
    environment.pop("BASH_ENV", None)
    environment.pop("ENV", None)
    return environment


def mmuse_options(args):
    prefixes = (
        "--correction-",
        "--enable-correction-head",
        "--selector-correction-",
        "--dflash-context-",
        "--dflash-block-",
        "--dflash-gated-",
        "--dflash2-",
    )
    return [arg for arg in args if arg.replace("--no-", "--", 1).startswith(prefixes)]


@unittest.skipUnless(BASH, "Bash is required for shell wiring tests")
class QwenTrainingScriptTests(unittest.TestCase):
    def run_script(self, model_type=None, replacements=()):
        # A real source path is needed for BASH_SOURCE-relative helper imports.
        recipe = SCRIPT.read_bytes().decode()
        for original, replacement in replacements:
            self.assertEqual(recipe.count(original), 1)
            recipe = recipe.replace(original, replacement)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "arguments"
            environment_capture = root / "environment"
            script_path = SCRIPT
            if replacements:
                recipe_dir = root / "edited recipe with spaces"
                recipe_dir.mkdir()
                script_path = recipe_dir / SCRIPT.name
                script_path.write_bytes(recipe.encode())  # Preserve raw EOLs.
                shutil.copytree(SCRIPT.parent / "common", recipe_dir / "common")
            script = f"source {shlex.quote(script_path.as_posix())}\n"
            environment = _shell_environment(
                {
                    "FIXTURE_ROOT": root.as_posix(),
                    "CAPTURE": capture.as_posix(),
                    "ENV_CAPTURE": environment_capture.as_posix(),
                }
            )
            environment.pop("SPECULATOR_TYPE", None)
            if model_type is not None:
                environment["SPECULATOR_TYPE"] = model_type
            result = subprocess.run(  # noqa: S603 -- Fixed recipe, fake launch commands.
                [BASH, "--noprofile", "--norc"],
                input=STUBS + script + "\nwait\n",
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
                check=False,
            )
            self.assertFalse((root / "output").exists())
            args = capture.read_text().splitlines() if capture.exists() else []
            self.captured_environment = (
                dict(
                    line.split("=", 1)
                    for line in environment_capture.read_text().splitlines()
                )
                if environment_capture.exists()
                else {}
            )
        return result, args

    def assert_option(self, args, option, expected):
        self.assertEqual(args.count(option), 1)
        self.assertEqual(args[args.index(option) + 1], expected)

    def test_default_and_explicit_dspark_only_pass_baseline_options(self):
        for model_type in (None, "dspark"):
            with self.subTest(model_type=model_type):
                result, args = self.run_script(model_type)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("torchrun", args)
                self.assertIn("scripts/train.py", args)
                self.assert_option(args, "--speculator-type", "dspark")
                self.assert_option(args, "--markov-rank", "256")
                self.assert_option(args, "--lr", "6e-4")
                self.assertIn("--enable-confidence-head", args)
                self.assertIn("--confidence-head-with-markov", args)
                self.assertEqual(mmuse_options(args), [])
                self.assertNotIn("", args)

    def test_mmuse_keeps_default_enhancement_configuration(self):
        result, args = self.run_script("mmuse")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_option(args, "--speculator-type", "mmuse")
        for option, value in (
            ("--correction-output-mode", "hidden"),
            ("--correction-hidden-size", "512"),
            ("--correction-rank", "256"),
            ("--correction-hidden-aux-weight", "0.1"),
            ("--correction-markov-gate-bias", "-2.0"),
            ("--selector-correction-feedback", "static"),
            ("--dflash2-conv-kernel-size", "2"),
            ("--dflash2-selector-rank", "256"),
            ("--dflash2-selector-top-k", "16"),
            ("--dflash2-selector-loss-weight", "1.0"),
        ):
            self.assert_option(args, option, value)
        self.assertEqual(len(mmuse_options(args)), 27)
        self.assertIn("--no-correction-hidden-feedback", args)
        self.assertIn("--no-dflash-gated-layer-fusion", args)
        self.assertIn("--no-dflash2-candidate-selector", args)
        self.assertIn("--dflash2-selector-greedy", args)
        self.assertNotIn("--dflash2-selector-global", args)
        self.assertNotIn("--enable-correction-head", args)
        self.assertNotIn("", args)

    def test_legacy_muse_alias_keeps_identical_enhancement_arguments(self):
        result, canonical = self.run_script("mmuse")
        self.assertEqual(result.returncode, 0, result.stderr)
        result, legacy = self.run_script("muse")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(legacy, canonical)

    def test_shared_ascend_environment_is_exported_to_launch_command(self):
        for model_type in ("dspark", "mmuse"):
            with self.subTest(model_type=model_type):
                result, _ = self.run_script(model_type)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.captured_environment, ASCEND_ENVIRONMENT)

    def run_helper_source(self):
        # Importing the helper must not export settings or alter its caller state.
        names = " ".join(ASCEND_ENVIRONMENT)
        program = (
            'set -eu\nfixture_pwd="$PWD"\nfixture_options="$-"\n'
            f"source {shlex.quote(COMMON_ENV.as_posix())}\n"
            '[[ "$PWD" == "$fixture_pwd" && "$-" == "$fixture_options" ]]\n'
            "declare -F configure_ascend_training_env > /dev/null\n"
            f"for name in {names} NO_PROXY; do\n"
            '  printf "%s=%s\\n" "$name" "${!name}"\ndone\n'
        )
        with tempfile.TemporaryDirectory(
            prefix="helper caller with spaces "
        ) as temporary:
            return subprocess.run(  # noqa: S603 -- Import function definitions only.
                [BASH, "--noprofile", "--norc"],
                input=program,
                cwd=temporary,
                env=_shell_environment(
                    {
                        **dict.fromkeys(ASCEND_ENVIRONMENT, "unchanged"),
                        "NO_PROXY": "proxy-unchanged",
                    }
                ),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

    def test_sourcing_environment_helper_only_defines_function(self):
        result = self.run_helper_source()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            dict(line.split("=", 1) for line in result.stdout.splitlines()),
            {
                **dict.fromkeys(ASCEND_ENVIRONMENT, "unchanged"),
                "NO_PROXY": "proxy-unchanged",
            },
        )

    def test_inherited_shell_startup_is_not_executed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            startup, marker = root / "startup.sh", root / "startup-ran"
            startup.write_bytes(b'printf "startup\\n" >> "$STARTUP_MARKER"\n')
            with patch.dict(
                os.environ,
                {
                    "BASH_ENV": startup.as_posix(),
                    "ENV": startup.as_posix(),
                    "STARTUP_MARKER": marker.as_posix(),
                },
            ):
                # A harmless control proves this Bash really honors the injected path.
                control = subprocess.run(  # noqa: S603 -- Only a temporary printf hook.
                    [BASH, "--noprofile", "--norc"],
                    input=":\n",
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(control.returncode, 0, control.stderr)
                self.assertEqual(marker.read_text(), "startup\n")
                marker.unlink()
                result, args = self.run_script("mmuse")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("torchrun", args)
                self.assertFalse(marker.exists())
                result = self.run_helper_source()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(marker.exists())

    def test_mmuse_preserves_enabled_correction_and_selector_settings(self):
        result, args = self.run_script(
            "mmuse",
            (
                (
                    "CORRECTION_HEAD_ARGS=()",
                    "CORRECTION_HEAD_ARGS=(--enable-correction-head)",
                ),
                ('CORRECTION_OUTPUT_MODE="hidden"', 'CORRECTION_OUTPUT_MODE="logits"'),
                ("--no-dflash2-candidate-selector)", "--dflash2-candidate-selector)"),
                (
                    "DFLASH2_SELECTOR_SEARCH_MODE=greedy",
                    "DFLASH2_SELECTOR_SEARCH_MODE=global",
                ),
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_option(args, "--correction-output-mode", "logits")
        self.assertIn("--enable-correction-head", args)
        self.assertIn("--dflash2-candidate-selector", args)
        self.assertIn("--dflash2-selector-global", args)
        self.assertNotIn("--no-dflash2-candidate-selector", args)
        self.assertNotIn("--dflash2-selector-greedy", args)

    def test_selector_validation_only_applies_to_mmuse(self):
        for replacements in (
            (
                (
                    "DFLASH2_SELECTOR_SEARCH_MODE=greedy",
                    "DFLASH2_SELECTOR_SEARCH_MODE=invalid",
                ),
            ),
            (
                (
                    "SELECTOR_CORRECTION_FEEDBACK=static",
                    "SELECTOR_CORRECTION_FEEDBACK=invalid",
                ),
            ),
            (
                (
                    "DFLASH2_SELECTOR_SEARCH_MODE=greedy",
                    "DFLASH2_SELECTOR_SEARCH_MODE=global",
                ),
                (
                    "SELECTOR_CORRECTION_FEEDBACK=static",
                    "SELECTOR_CORRECTION_FEEDBACK=corrected",
                ),
            ),
        ):
            with self.subTest(replacements=replacements):
                result, args = self.run_script("dspark", replacements)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(mmuse_options(args), [])
                result, args = self.run_script("mmuse", replacements)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(args, [])


if __name__ == "__main__":
    unittest.main()
