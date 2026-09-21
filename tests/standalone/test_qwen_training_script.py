"""Exercise Qwen launch arguments without starting training or creating run outputs."""

# ruff: noqa: PT009 -- Also runnable without pytest/torch.

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/train/dspark_qwen3_8b_trainer.sh"
if os.name == "nt":
    BASH = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
    BASH = str(BASH) if BASH.is_file() else None
else:
    BASH = shutil.which("bash")

STUBS = r"""
# Redirect log/PID writes into the fixture, without creating the recipe's output.
mkdir() { LOG_DIR="$FIXTURE_ROOT"; }
date() { printf 'fixture\n'; }
nohup() { printf '%s\n' "$@" > "$CAPTURE"; }
"""


def muse_options(args):
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
        # Source the real file normally; edited-recipe cases preserve its raw EOLs.
        script = (
            SCRIPT.read_bytes().decode()
            if replacements
            else f"source {shlex.quote(SCRIPT.as_posix())}\n"
        )
        for original, replacement in replacements:
            self.assertEqual(script.count(original), 1)
            script = script.replace(original, replacement)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "arguments"
            environment = {
                **os.environ,
                "FIXTURE_ROOT": root.as_posix(),
                "CAPTURE": capture.as_posix(),
            }
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
                self.assertEqual(muse_options(args), [])
                self.assertNotIn("", args)

    def test_muse_keeps_default_enhancement_configuration(self):
        result, args = self.run_script("muse")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_option(args, "--speculator-type", "muse")
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
        self.assertEqual(len(muse_options(args)), 27)
        self.assertIn("--no-correction-hidden-feedback", args)
        self.assertIn("--no-dflash-gated-layer-fusion", args)
        self.assertIn("--no-dflash2-candidate-selector", args)
        self.assertIn("--dflash2-selector-greedy", args)
        self.assertNotIn("--dflash2-selector-global", args)
        self.assertNotIn("--enable-correction-head", args)
        self.assertNotIn("", args)

    def test_muse_preserves_enabled_correction_and_selector_settings(self):
        result, args = self.run_script(
            "muse",
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

    def test_selector_validation_only_applies_to_muse(self):
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
                self.assertEqual(muse_options(args), [])
                result, args = self.run_script("muse", replacements)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(args, [])


if __name__ == "__main__":
    unittest.main()
