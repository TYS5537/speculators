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
DFLASH2_TRAINER = "train_dflash2.sh"
ALL_TRAINERS = (*TRAINERS, DFLASH2_TRAINER)
HTTP_TRAINER = "train_mmuse_legacy_optimizer_http.sh"
HTTP_SERVER = "server_http.sh"
LAUNCH_MARKER = "# ============ Launch ============"


def recipe_source(name, settings=None):
    """Simulate editing settings at the top, without modifying repo files."""
    configuration, launch = (
        (RECIPES / name).read_text(encoding="utf-8").split(LAUNCH_MARKER, 1)
    )
    for key, value in (settings or {}).items():
        quoted = (
            "(" + " ".join(shlex.quote(item) for item in value) + ")"
            if isinstance(value, (list, tuple))
            else shlex.quote(value)
        )
        configuration, count = re.subn(
            rf"^{re.escape(key)}=.*$",
            lambda match, key=key, quoted=quoted: f"{key}={quoted}",
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
        if name in (*ALL_TRAINERS, HTTP_TRAINER)
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
    def test_standalone_scripts_without_shared_configuration(self):
        self.assertEqual(
            {path.name for path in RECIPES.glob("*.sh")},
            {*ALL_TRAINERS, "server.sh", HTTP_TRAINER, HTTP_SERVER},
        )
        for name in (*ALL_TRAINERS, "server.sh"):
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
        for name in ALL_TRAINERS:
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

    def test_http_trainer_preserves_legacy_objective_and_connects_both_services(self):
        args = training_arguments(HTTP_TRAINER)
        original = training_arguments(TRAINERS[1])
        for flag in (
            "--optimizer",
            "--muon-parameter-policy",
            "--lr",
            "--muon-lr",
            "--weight-decay",
            "--muon-weight-decay",
            "--loss-fn",
        ):
            self.assertEqual(
                args[args.index(flag) + 1], original[original.index(flag) + 1]
            )
        for flag, value in {
            "--hidden-states-backend": "http",
            "--vllm-endpoint": "http://127.0.0.1:8001/v1",
            "--hs-http-endpoints": "http://80.5.5.101:8002",
            "--block-size": "15",
            "--max-anchors": "512",
            "--on-generate": "delete",
        }.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--fsdp-shard", args)
        self.assertIn("--activation-checkpointing", args)
        self.assertNotIn("--target-hidden-state-format", args)

    def test_http_server_uses_local_connector_explicit_bf16_and_teacher_slot(self):
        args = preview_command(HTTP_SERVER)
        for flag in ("--include-last-layer", "--provenance-dir"):
            self.assertIn(flag, args)
        for flag, value in {
            "--hidden-states-backend": "file",
            "--dtype": "bfloat16",
            "--host": "0.0.0.0",  # noqa: S104 -- Expected LAN bind in a fixture.
            "--served-model-name": "qwen3-4b-http",
        }.items():
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_http_missing_host_or_credential_fails_before_launch(self):
        result = run_recipe(HTTP_TRAINER, settings={"TARGET_HOST": "CHANGE_ME"})
        self.assertEqual(result.returncode, 2)
        self.assertIn("TARGET_HOST", result.stderr)
        for name in (HTTP_SERVER, HTTP_TRAINER):
            for token in ("", "short"):
                with self.subTest(script=name, token=token):
                    result = run_recipe(name, settings={"HTTP_TOKEN": token})
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("HTTP_TOKEN", result.stderr)

    def test_http_two_nodes_join_one_job_with_six_workers_each(self):
        endpoints = ["http://192.0.2.10:8002", "http://192.0.2.11:8002"]
        for node in (0, 1):
            command = preview_command(
                HTTP_TRAINER,
                TARGET_HOST="127.0.0.1",
                NUM_TRAIN_NODES="2",
                TRAIN_NODE_RANK=str(node),
                TRAIN_MASTER_ADDR="192.0.2.10",
                HS_HTTP_ENDPOINTS=endpoints,
            )
            for flag, value in {
                "--nnodes": "2",
                "--node_rank": str(node),
                "--nproc_per_node": "6",
                "--master_addr": "192.0.2.10",
                "--master_port": "34655",
                "--max_restarts": "0",
                "--vllm-endpoint": "http://127.0.0.1:8001/v1",
            }.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertNotIn("--standalone", command)
            self.assertIn("ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7", command)
            start = command.index("--hs-http-endpoints") + 1
            self.assertEqual(command[start : start + 2], endpoints)
            self.assertEqual(
                command[command.index("--rdzv_id") + 1],
                command[command.index("--run-name") + 1],
            )

    def test_http_multinode_rejects_invalid_topology_before_contacting_servers(self):
        settings = {
            "TARGET_HOST": "127.0.0.1",
            "NUM_TRAIN_NODES": "2",
            "TRAIN_NODE_RANK": "1",
            "TRAIN_MASTER_ADDR": "192.0.2.10",
            "TRAIN_LOCAL_IP": "192.0.2.11",
            "TRAIN_IFNAME": "eth0",
            "HS_HTTP_ENDPOINTS": [
                "http://192.0.2.10:8002",
                "http://192.0.2.11:8002",
            ],
        }
        for invalid, message in (
            ({"NUM_TRAIN_NODES": "0"}, "NUM_TRAIN_NODES"),
            ({"TRAIN_NODE_RANK": "2"}, "TRAIN_NODE_RANK"),
            ({"TRAIN_NODE_RANK": "-1"}, "TRAIN_NODE_RANK"),
            ({"TRAIN_MASTER_ADDR": "127.0.0.1"}, "TRAIN_MASTER_ADDR"),
            ({"TRAIN_LOCAL_IP": "CHANGE_ME"}, "TRAIN_LOCAL_IP"),
            ({"TRAIN_IFNAME": "lo"}, "TRAIN_IFNAME"),
            ({"HS_HTTP_ENDPOINTS": ["http://192.0.2.10:8002"]}, "HS_HTTP_ENDPOINTS"),
            ({"TRAIN_MASTER_PORT": "65536"}, "TRAIN_MASTER_PORT"),
        ):
            with self.subTest(invalid=invalid):
                result = run_recipe(
                    HTTP_TRAINER,
                    settings={**settings, **invalid},
                    prelude=(
                        "export SPECULATORS_HS_HTTP_TOKEN="
                        "fixture-only-0123456789abcdef0123456789"
                    ),
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)

    def test_http_auto_node_detection_rejects_missing_or_ambiguous_addresses(self):
        for addresses, expected_count in (
            ("1: lo inet 127.0.0.1/8 scope host lo", 0),
            (
                "2: bond0 inet 80.5.5.101/24 scope global bond0\n"
                "3: bond1 inet 80.5.5.137/24 scope global bond1",
                2,
            ),
        ):
            with self.subTest(addresses=addresses):
                result = run_recipe(
                    HTTP_TRAINER,
                    prelude=(
                        "export SPECULATORS_HS_HTTP_TOKEN="
                        "fixture-only-0123456789abcdef0123456789\n"
                        f"ip() {{ printf '%s\\n' {shlex.quote(addresses)}; }}"
                    ),
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(
                    f"exactly one local TRAIN_NODE_IPS address; found {expected_count}",
                    result.stderr,
                )

    def test_http_auto_node_detection_rejects_failed_query_and_conflicting_settings(
        self,
    ):
        credential = (
            "export SPECULATORS_HS_HTTP_TOKEN=fixture-only-0123456789abcdef0123456789\n"
        )
        result = run_recipe(HTTP_TRAINER, prelude=credential + "ip() { return 1; }")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Cannot read local IPv4 interfaces", result.stderr)
        for settings, message in (
            ({"TRAIN_NODE_RANK": "0"}, "conflict with the detected local node"),
            ({"TRAIN_LOCAL_IP": "80.5.5.101"}, "conflict with the detected local node"),
            ({"TRAIN_IFNAME": "eth0"}, "conflict with the detected local node"),
            ({"TRAIN_NODE_IPS": ["80.5.5.137"]}, "one ordered IP per training node"),
        ):
            with self.subTest(settings=settings):
                result = run_recipe(
                    HTTP_TRAINER,
                    settings=settings,
                    prelude=credential
                    + (
                        "ip() { printf '%s\\n' "
                        "'2: bond0 inet 80.5.5.137/24 scope global bond0'; }"
                    ),
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(message, result.stderr)

    def test_http_multinode_launch_detects_local_network_and_checks_both_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            hosts = (("80.5.5.101", "bond0"), ("80.5.5.137", "enp1s0@if7"))
            for node, (local_ip, address_interface) in enumerate(hosts):
                interface = address_interface.split("@")[0]
                output = root / f"node{node}"
                settings = {
                    "DATA_PATH": data.as_posix(),
                    "OUTPUT_DIR": output.as_posix(),
                }
                addresses = (
                    "1: lo inet 127.0.0.1/8 scope host lo\n"
                    f"2: {address_interface} inet {local_ip}/24 "
                    f"scope global {interface}"
                )
                prelude = (
                    f"ip() {{ printf '%s\\n' {shlex.quote(addresses)}; }}\n"
                    r"""
export SPECULATORS_HS_HTTP_TOKEN=fixture-only-0123456789abcdef0123456789
curl() { return 0; }
python() { mkdir -p "$OUTPUT_DIR"; printf '%s\0' "$@" > "$OUTPUT_DIR/preflight.argv"; }
nohup() {
    printf '%s\n' "$GLOO_SOCKET_IFNAME" "$HCCL_SOCKET_IFNAME" "$HCCL_IF_IP" \
        > "$LOG_DIR/network.txt"
    printf '%s\0' "$@"
}
"""
                )
                result = run_recipe(
                    HTTP_TRAINER, settings=settings, prelude=prelude, trailer="wait"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Global training workers: 12", result.stdout)
                self.assertIn(
                    f"Training node {node}: {local_ip} on {interface}; "
                    "master 80.5.5.101:34655",
                    result.stdout,
                )
                logdir = output / "logs" / f"node{node}"
                log = next(logdir.glob("train_*.log"))
                self.assertEqual(
                    log.read_text().rstrip("\0").split("\0"),
                    preview_command(
                        HTTP_TRAINER, **settings, TRAIN_NODE_RANK=str(node)
                    ),
                )
                self.assertEqual(
                    (logdir / "network.txt").read_text().splitlines(),
                    [interface, f"={interface}", local_ip],
                )
                preflight = (output / "preflight.argv").read_text().split("\0")
                for host_ip, _ in hosts:
                    self.assertIn(f"http://{host_ip}:8002", preflight)
                self.assertNotIn("fixture-only", (logdir / "topology.txt").read_text())
                repeated = run_recipe(HTTP_TRAINER, settings=settings, prelude=prelude)
                self.assertEqual(repeated.returncode, 2)
                self.assertIn("already used", repeated.stderr)

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
            for name in ALL_TRAINERS:
                missing = run_recipe(name, settings=settings)
                self.assertEqual(missing.returncode, 2)
                self.assertIn("Arrow data not found", missing.stderr)
            (output / "checkpoints").mkdir(parents=True)
            settings["DATA_PATH"] = data.as_posix()
            for name in ALL_TRAINERS:
                existing = run_recipe(name, settings=settings)
                self.assertEqual(existing.returncode, 2)
                self.assertIn("already used", existing.stderr)

    def test_training_launch_writes_log_pid_and_uses_edited_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "prepared data"
            data.mkdir()
            for name in ALL_TRAINERS:
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
