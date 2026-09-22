"""Exercise complete online recipes with fake commands, never training or HTTP."""

# ruff: noqa: PT009 -- Also runnable without pytest or the model dependencies.

import os
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_EXAMPLES = ROOT / "examples/train"
RECIPES = {
    "ascend": "dspark_qwen3_8b_sharegpt_online_ascend.sh",
    "cuda": "dspark_qwen3_0_6b_sharegpt_online.sh",
}
COMMON_FILES = ("dspark_online_args.sh", "ascend_training_env.sh")
if os.name == "nt":
    BASH = Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Git/bin/bash.exe"
    BASH = str(BASH) if BASH.is_file() else None
else:
    BASH = shutil.which("bash")

PERFORMANCE_ENV = {
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
SEED_ENV = {
    **{name: f"caller-{name.lower()}" for name in PERFORMANCE_ENV},
    "NO_PROXY": "caller-upper-proxy",
    "no_proxy": "caller-lower-proxy",
    "CUDA_VISIBLE_DEVICES": "caller-cuda",
    "ASCEND_RT_VISIBLE_DEVICES": "caller-ascend",
}
SEED_SHELL = "\n".join(
    f"export {name}={shlex.quote(value)}" for name, value in SEED_ENV.items()
)
ASCEND_NO_PROXY = "localhost,127.0.0.1,80.5.5.45,80.5.5.44,80.5.5.54"

STUBS = r"""
capture_call() {
    local kind="$1"
    shift
    printf '%s\0' "$@" > "$CAPTURE_DIR/$kind.argv"
    command env > "$CAPTURE_DIR/$kind.env"
    printf '%s\n' "$PWD" > "$CAPTURE_DIR/$kind.cwd"
}
python() {
    case "$1" in
        scripts/prepare_data.py) capture_call prepare "$@" ;;
        scripts/launch_vllm.py) capture_call server "$@" ;;
        *) printf 'Unexpected Python command: %s\n' "$*" >&2; return 97 ;;
    esac
}
# Keep env assignments local to this invocation and execute only the fake commands.
env() (
    while [[ "$1" == *=* ]]; do
        export "$1"
        shift
    done
    "$@"
)
torchrun() {
    capture_call training "$@"
    printf '%s\0' "$VLLM_PID" > "$CAPTURE_DIR/owned-pid.argv"
    return "$TRAIN_EXIT_STATUS"
}
curl_calls=0
curl() {
    curl_calls=$((curl_calls + 1))
    printf '%s\0' "$@" > "$CAPTURE_DIR/curl-$curl_calls.argv"
    (( curl_calls > 1 ))
}
sleep() { printf '%s\0' "$@" >> "$CAPTURE_DIR/sleep.argv"; }
kill() { printf '%s\0' "$@" >> "$CAPTURE_DIR/kill.argv"; }
wait() {
    printf '%s\0' "$@" >> "$CAPTURE_DIR/wait.argv"
    builtin wait "$@"
}
"""


def recipe_values(kind):
    """Independent golden values, not parsed from the shell implementation."""
    return {
        "MODEL": "Qwen/Qwen3-8B" if kind == "ascend" else "Qwen/Qwen3-0.6B",
        "DATASET": "sharegpt",
        "OUTPUT_DIR": (
            "./output/dspark_qwen3_8b_sharegpt_ascend"
            if kind == "ascend"
            else "./output/dspark_qwen3_0_6b_sharegpt"
        ),
        "VLLM_PORT": "8000",
        "MAX_SAMPLES": "5000",
        "SEQ_LENGTH": "8192" if kind == "ascend" else "4096",
        "EPOCHS": "10",
        "LR": "3e-4",
        "SPECULATOR_TYPE": "dspark",
        "BLOCK_SIZE": "7",
        "MAX_ANCHORS": "3072",
        "NUM_LAYERS": "5",
        "DRAFT_VOCAB_SIZE": "32000",
        "TARGET_LAYER_IDS": "2 18 33" if kind == "ascend" else "2 14 25",
        "MARKOV_RANK": "256",
        "MARKOV_HEAD_TYPE": "vanilla",
        "LOSS_FN": '{"ce": 0.1, "tv": 0.9}',
        "CONFIDENCE_HEAD_ALPHA": "1.0",
        "CONFIDENCE_LOSS_WEIGHTING": "match-draft",
        "DRAFT_ATTN_IMPL": "eager",
    }


def expected_train_arguments(values, attention_args=()):
    return [
        "--verifier-name-or-path",
        values["MODEL"],
        "--data-path",
        values["OUTPUT_DIR"],
        "--vllm-endpoint",
        f"http://localhost:{values['VLLM_PORT']}/v1",
        "--save-path",
        f"{values['OUTPUT_DIR']}/checkpoints",
        "--draft-vocab-size",
        values["DRAFT_VOCAB_SIZE"],
        "--epochs",
        values["EPOCHS"],
        "--lr",
        values["LR"],
        "--total-seq-len",
        values["SEQ_LENGTH"],
        "--speculator-type",
        values["SPECULATOR_TYPE"],
        "--block-size",
        values["BLOCK_SIZE"],
        "--max-anchors",
        values["MAX_ANCHORS"],
        "--num-layers",
        values["NUM_LAYERS"],
        *attention_args,
        "--target-layer-ids",
        *values["TARGET_LAYER_IDS"].split(),
        "--markov-rank",
        values["MARKOV_RANK"],
        "--markov-head-type",
        values["MARKOV_HEAD_TYPE"],
        "--enable-confidence-head",
        "--confidence-head-with-markov",
        "--loss-fn",
        values["LOSS_FN"],
        "--confidence-head-alpha",
        values["CONFIDENCE_HEAD_ALPHA"],
        "--confidence-loss-weighting",
        values["CONFIDENCE_LOSS_WEIGHTING"],
        "--on-missing",
        "generate",
        "--on-generate",
        "delete",
    ]


@unittest.skipUnless(BASH, "Bash is required for shell wiring tests")
class DSparkOnlineScriptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run_number = 0

    def make_tree(self):
        self.run_number += 1
        fixture = self.root / str(self.run_number)
        recipes = fixture / "recipe tree with spaces" / "examples/train"
        common = recipes / "common"
        common.mkdir(parents=True)
        for name in RECIPES.values():
            shutil.copyfile(TRAIN_EXAMPLES / name, recipes / name)
        for name in COMMON_FILES:
            shutil.copyfile(TRAIN_EXAMPLES / "common" / name, common / name)
        working = fixture / "unrelated working directory"
        working.mkdir()
        captures = fixture / "captures"
        captures.mkdir()
        return recipes, working, captures

    def run_bash(self, shell, working, captures, training_status=0):
        environment = {
            **os.environ,
            "CAPTURE_DIR": captures.as_posix(),
            "TRAIN_EXIT_STATUS": str(training_status),
        }
        environment.pop("BASH_ENV", None)
        environment.pop("ENV", None)
        return subprocess.run(  # noqa: S603 -- Fixed fixtures with fake commands.
            [BASH, "--noprofile", "--norc"],
            input=shell,
            cwd=working,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )

    def arguments(self, captures, name):
        payload = (captures / f"{name}.argv").read_bytes()
        self.assertTrue(payload.endswith(b"\0"))
        return payload[:-1].decode("utf-8").split("\0")

    @staticmethod
    def environment(captures, name):
        result = {}
        for line in (captures / f"{name}.env").read_text().splitlines():
            key, _, value = line.partition("=")
            if key in SEED_ENV:
                result[key] = value
        return result

    def run_recipe(self, kind, overrides=None, training_status=0):
        recipes, working, captures = self.make_tree()
        recipe = recipes / RECIPES[kind]
        values = recipe_values(kind)
        source = recipe.read_bytes().decode("utf-8")
        for name, value in (overrides or {}).items():
            old = values[name]
            quote = "'" if name == "LOSS_FN" else '"'
            original = f"{name}={quote}{old}{quote}"
            self.assertEqual(source.count(original), 1)
            source = source.replace(original, f"{name}={quote}{value}{quote}")
            values[name] = value
        # Preserve the recipe's real line endings while editing only fixture values.
        recipe.write_bytes(source.encode("utf-8"))
        shell = (
            SEED_SHELL
            + "\n"
            + STUBS
            + '\nprintf "%s\\n" "$PWD" > "$CAPTURE_DIR/initial.cwd"\n'
            + f"source {shlex.quote(recipe.as_posix())}\n"
        )
        result = self.run_bash(shell, working, captures, training_status)
        self.assertEqual(result.returncode, training_status, result.stderr)
        self.assertFalse((working / "output").exists())
        self.assertFalse((working / "output with spaces").exists())
        return result, captures, values

    def assert_recipe_contract(self, kind, captures, values):
        self.assertEqual(
            self.arguments(captures, "prepare"),
            [
                "scripts/prepare_data.py",
                "--model",
                values["MODEL"],
                "--data",
                values["DATASET"],
                "--output",
                values["OUTPUT_DIR"],
                "--max-samples",
                values["MAX_SAMPLES"],
                "--seq-length",
                values["SEQ_LENGTH"],
                "--disable-thinking",
            ],
        )
        self.assertEqual(
            self.arguments(captures, "server"),
            [
                "scripts/launch_vllm.py",
                values["MODEL"],
                "--target-layer-ids",
                *values["TARGET_LAYER_IDS"].split(),
                "--",
                "--port",
                values["VLLM_PORT"],
                *(
                    ["--enforce-eager", "--data-parallel-size", "4"]
                    if kind == "ascend"
                    else []
                ),
            ],
        )
        attention = (
            ["--draft-attn-impl", values["DRAFT_ATTN_IMPL"]] if kind == "ascend" else []
        )
        self.assertEqual(
            self.arguments(captures, "training"),
            [
                "--standalone",
                "--nproc_per_node",
                "4" if kind == "ascend" else "1",
                "scripts/train.py",
                *expected_train_arguments(values, attention),
            ],
        )
        probe = (
            [
                "--noproxy",
                "localhost,127.0.0.1",
                "-sf",
                f"http://localhost:{values['VLLM_PORT']}/v1/models",
            ]
            if kind == "ascend"
            else ["-sf", f"http://localhost:{values['VLLM_PORT']}/health"]
        )
        self.assertEqual(self.arguments(captures, "curl-1"), probe)
        self.assertEqual(self.arguments(captures, "curl-2"), probe)
        self.assertFalse((captures / "curl-3.argv").exists())
        self.assertEqual(
            self.arguments(captures, "sleep"), ["5" if kind == "ascend" else "2"]
        )

        expected_environment = dict(SEED_ENV)
        if kind == "ascend":
            expected_environment.update(
                PERFORMANCE_ENV, NO_PROXY=ASCEND_NO_PROXY, no_proxy=ASCEND_NO_PROXY
            )
        self.assertEqual(self.environment(captures, "prepare"), expected_environment)
        device = (
            "ASCEND_RT_VISIBLE_DEVICES" if kind == "ascend" else "CUDA_VISIBLE_DEVICES"
        )
        self.assertEqual(
            self.environment(captures, "server"),
            {**expected_environment, device: "0,1,2,3" if kind == "ascend" else "0"},
        )
        self.assertEqual(
            self.environment(captures, "training"),
            {**expected_environment, device: "4,5,6,7" if kind == "ascend" else "1"},
        )
        initial_cwd = (captures / "initial.cwd").read_text()
        for stage in ("prepare", "server", "training"):
            self.assertEqual((captures / f"{stage}.cwd").read_text(), initial_cwd)
        owned_pid = self.arguments(captures, "owned-pid")
        self.assertEqual(len(owned_pid), 1)
        self.assertRegex(owned_pid[0], r"^[1-9][0-9]*$")
        self.assertEqual(self.arguments(captures, "kill"), owned_pid)
        self.assertEqual(self.arguments(captures, "wait"), owned_pid)

    def test_baseline_scripts_preserve_complete_launch_contract(self):
        for kind in RECIPES:
            with self.subTest(kind=kind):
                result, captures, values = self.run_recipe(kind)
                self.assert_recipe_contract(kind, captures, values)
                self.assertIn("Done. Checkpoints saved", result.stdout)
                self.assertIn("Stopping vLLM server...", result.stdout)

    def test_custom_recipe_preserves_spaces_json_and_target_expansion(self):
        for kind in RECIPES:
            with self.subTest(kind=kind):
                overrides = {
                    "MODEL": "/fixture/model weights/Qwen model",
                    "DATASET": "/fixture/data sets/share gpt.json",
                    "OUTPUT_DIR": "./output with spaces/run 01",
                    "LOSS_FN": '{"ce": 0.25, "tv": 0.75}',
                    "TARGET_LAYER_IDS": "3 11 19",
                }
                if kind == "ascend":
                    overrides["DRAFT_ATTN_IMPL"] = "sdpa"
                _, captures, values = self.run_recipe(kind, overrides)
                self.assert_recipe_contract(kind, captures, values)
                self.assertEqual(
                    self.arguments(captures, "training").count(values["LOSS_FN"]), 1
                )

    def test_training_failure_still_cleans_up_only_owned_server(self):
        for kind in RECIPES:
            with self.subTest(kind=kind):
                result, captures, values = self.run_recipe(kind, training_status=23)
                self.assert_recipe_contract(kind, captures, values)
                self.assertNotIn("Done. Checkpoints saved", result.stdout)
                self.assertIn("Stopping vLLM server...", result.stdout)

    def test_sourcing_helpers_only_defines_functions(self):
        recipes, working, captures = self.make_tree()
        shell = (
            "set -euo pipefail\n"
            + SEED_SHELL
            + '\nDSPARK_TRAIN_ARGS=(sentinel "keep this argument")\n'
            + 'trap ":" EXIT\n'
            + 'command env > "$CAPTURE_DIR/before.env"\n'
            + 'printf "%s\\n" "$PWD" > "$CAPTURE_DIR/before.cwd"\n'
            + 'set +o > "$CAPTURE_DIR/before.options"\n'
            + 'trap -p > "$CAPTURE_DIR/before.traps"\n'
            + "\n".join(
                f"source {shlex.quote((recipes / 'common' / name).as_posix())}"
                for name in COMMON_FILES
            )
            + '\ncommand env > "$CAPTURE_DIR/after.env"\n'
            + 'printf "%s\\n" "$PWD" > "$CAPTURE_DIR/after.cwd"\n'
            + 'set +o > "$CAPTURE_DIR/after.options"\n'
            + 'trap -p > "$CAPTURE_DIR/after.traps"\n'
            + 'printf "%s\\0" "${DSPARK_TRAIN_ARGS[@]}" > "$CAPTURE_DIR/array.argv"\n'
            + "declare -F build_dspark_online_train_args > /dev/null\n"
            + "declare -F configure_ascend_training_env > /dev/null\n"
        )
        result = self.run_bash(shell, working, captures)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.environment(captures, "before"), SEED_ENV)
        self.assertEqual(self.environment(captures, "after"), SEED_ENV)
        for suffix in ("cwd", "options", "traps"):
            self.assertEqual(
                (captures / f"before.{suffix}").read_text(),
                (captures / f"after.{suffix}").read_text(),
            )
        self.assertEqual(
            self.arguments(captures, "array"), ["sentinel", "keep this argument"]
        )

    def test_argument_builder_replaces_instead_of_accumulating(self):
        recipes, working, captures = self.make_tree()
        first = recipe_values("ascend")
        second = {
            **first,
            "MODEL": "/other model",
            "TARGET_LAYER_IDS": "4 12",
            "LR": "9e-4",
        }
        helper = recipes / "common/dspark_online_args.sh"
        shell = (
            "set -euo pipefail\n"
            f"source {shlex.quote(helper.as_posix())}\n"
            + "\n".join(f"{name}={shlex.quote(value)}" for name, value in first.items())
            + "\nDSPARK_TRAIN_ARGS=(stale)\n"
            + "build_dspark_online_train_args --draft-attn-impl eager\n"
            + 'printf "%s\\0" "${DSPARK_TRAIN_ARGS[@]}" > "$CAPTURE_DIR/first.argv"\n'
            + "\n".join(
                f"{name}={shlex.quote(value)}" for name, value in second.items()
            )
            + "\nbuild_dspark_online_train_args\n"
            + 'printf "%s\\0" "${DSPARK_TRAIN_ARGS[@]}" > "$CAPTURE_DIR/second.argv"\n'
        )
        result = self.run_bash(shell, working, captures)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            self.arguments(captures, "first"),
            expected_train_arguments(first, ["--draft-attn-impl", "eager"]),
        )
        self.assertEqual(
            self.arguments(captures, "second"), expected_train_arguments(second)
        )

    def test_bash_syntax(self):
        for relative in [
            *RECIPES.values(),
            *(f"common/{name}" for name in COMMON_FILES),
        ]:
            with self.subTest(script=relative):
                script = TRAIN_EXAMPLES / relative
                self.assertNotIn(b"\r\n", script.read_bytes())
                result = subprocess.run(  # noqa: S603 -- Syntax check only.
                    [BASH, "-n", script.as_posix()],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
