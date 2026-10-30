"""Real CPU model evaluation exercises the device-isolated worker protocol."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators_eval import leagle, parallel, persistent, provenance

from .test_leagle_offline_eval import make_real_models, save_fixture


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(83)
    torch.set_num_threads(1)


def result_contents(path):
    rows = json.loads((path / "summary.json").read_text())
    for row in rows:
        for name in ("elapsed_s", "requests_per_second", "output_tokens_per_second"):
            row.pop(name, None)
    artifacts = {
        str(file.relative_to(path)): file.read_text()
        for file in (path / "artifacts").rglob("*.jsonl")
    }
    return rows, artifacts


@pytest.mark.parametrize("temperature", [0, 1])
@pytest.mark.parametrize("optimized", [False, True])
def test_models_load_once_and_dataset_rng_matches_fresh_workers(
    tmp_path, monkeypatch, temperature, optimized
):
    args = save_fixture(tmp_path)
    args.check_greedy = temperature == 0
    args.temperature = temperature
    args.leagle_static_kv = optimized
    args.leagle_reuse_rollout_buffers = optimized
    args.leagle_static_target_kv = optimized
    args.leagle_selective_target_hidden_states = optimized
    args.leagle_share_output_head = optimized
    provenance.prepare_checkpoints(args)
    entrypoint = Path("scripts/evaluate/leagle_offline_eval.py")
    second = tmp_path / "second.jsonl"
    second.write_text('{"prompt":"9 2 1 5"}\n', encoding="utf-8")
    paths = [args.datasets_root, second]
    commands, jobs = [], []
    for index, dataset in enumerate(paths):
        cmd = parallel.worker_command(
            args,
            entrypoint=entrypoint,
            dataset_path=dataset,
            shard_index=0,
            num_shards=1,
            output_dir=tmp_path / f"reference-{index}",
        )
        leagle.run(leagle.build_parser().parse_args(cmd[2:]), entrypoint=entrypoint)
        cmd[cmd.index("--output-dir") + 1] = str(tmp_path / f"persistent-{index}")
        commands.append(cmd)
        jobs.append(
            {"argv": cmd[1:], "completion": str(tmp_path / f"job-{index}.done")}
        )
    loads = []
    original = leagle.load_models

    def load(options):
        loads.append(1)
        return original(options)

    monkeypatch.setattr(leagle, "load_models", load)
    monkeypatch.setattr(
        sys, "stdin", io.StringIO("".join(json.dumps(job) + "\n" for job in jobs))
    )
    initial = leagle.build_parser().parse_args(
        [*commands[0][2:], "--worker-job-stream"]
    )
    leagle.run(initial, entrypoint=entrypoint)
    assert loads == [1]
    for index, job in enumerate(jobs):
        directory = tmp_path / f"persistent-{index}"
        assert Path(job["completion"]).is_file()
        assert result_contents(directory) == result_contents(
            tmp_path / f"reference-{index}"
        )
        command = (directory / "eval_command.txt").read_text()
        assert str(paths[index]) in command
        assert "--worker-job-stream" not in command
        metadata = json.loads((directory / "eval_config.json").read_text())
        assert metadata["persistent_workers"]
        assert metadata["worker_rng_policy"] == "restore-post-load-per-dataset"
        assert metadata["checkpoints"] == args.checkpoint_identity


def test_parent_reuses_processes_preserves_nested_shards_and_cleans_up(
    tmp_path, monkeypatch
):
    args = save_fixture(tmp_path)
    datasets = tmp_path / "datasets"
    for folder in ("a", "b"):
        (datasets / folder).mkdir(parents=True)
        records = args.datasets_root.read_text() + '{"prompt":"3 1"}\n'
        (datasets / folder / "toy.jsonl").write_text(
            records if folder == "b" else '{"prompt":"5 9 2"}\n', encoding="utf-8"
        )
    args.datasets_root = datasets
    args.temperature, args.check_greedy = 1, False
    args.ascend_devices = "3,6"
    provenance.prepare_checkpoints(args)
    # CPU emulates isolated devices here; real NPU initialization is unavailable.
    entrypoint = tmp_path / "worker.py"
    root = Path(__file__).resolve().parents[3]
    entrypoint.write_text(
        "import os, sys, types\n"
        f"sys.path[:0] = {[str(root / 'src'), str(root / 'hs_connectors/src')]!r}\n"
        "if os.name == 'nt':\n"
        "    fcntl = types.ModuleType('fcntl')\n"
        "    fcntl.LOCK_SH, fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN = 1,2,4,8\n"
        "    def unavailable(*args, **kwargs):\n"
        "        raise NotImplementedError('POSIX locking is unavailable on Windows')\n"
        "    fcntl.flock = unavailable\n"
        "    sys.modules['fcntl'] = fcntl\n"
        "import torch\ntorch.set_num_threads(1)\n"
        "from pathlib import Path\nfrom speculators_eval.leagle import main\n"
        "main(entrypoint=Path(__file__))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    original = parallel.subprocess.Popen
    children = []

    def launch(command, **kwargs):
        child = original(command, **kwargs)
        children.append((child, kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"]))
        return child

    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    outputs = []
    for enabled in (False, True):
        args.persistent_workers = enabled
        args.output_dir = tmp_path / f"parallel-{enabled}"
        parallel.run_ascend_data_parallel(args, entrypoint=entrypoint)
        outputs.append(result_contents(args.output_dir))
    assert outputs[0] == outputs[1]
    assert [device for _, device in children] == ["3", "6"] * 3
    assert all(child.poll() == 0 for child, _ in children)
    assert [row["dataset"] for row in outputs[1][0]] == ["a/toy", "b/toy"]


@pytest.mark.parametrize("code", [0, 1])
def test_exited_worker_cannot_be_mistaken_for_a_completed_job(tmp_path, code):
    with persistent.PersistentWorkers() as pool:
        pool.pending[0] = tmp_path / "missing.done"
        child = SimpleNamespace(poll=lambda: code)
        with pytest.raises(RuntimeError, match="exited"):
            pool.wait([(0, tmp_path, child)], "toy")


def test_rng_snapshot_includes_accelerator_state(monkeypatch):
    calls = []
    backend = SimpleNamespace(
        get_rng_state=lambda device: torch.tensor([17]),
        set_rng_state=lambda state, device: calls.append((state.clone(), str(device))),
    )
    monkeypatch.setattr(torch, "cuda", backend)
    restore = persistent.capture_rng("cuda:0")
    expected = torch.rand(3)
    restore()
    torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)
    assert calls[0][1] == "cuda:0"
    assert calls[0][0].item() == 17


def test_persistent_flag_requires_parent_device_sharding(tmp_path):
    args = save_fixture(tmp_path)
    args.persistent_workers = True
    with pytest.raises(ValueError, match="parent"):
        leagle.validate_args(args)


@pytest.mark.parametrize("role", ["target", "draft"])
def test_mutable_rotary_frequencies_cannot_leak_between_datasets(role):
    target, draft = make_real_models()
    persistent.validate_worker_models(target, draft)
    rotary = target.model.rotary_emb if role == "target" else draft.rotary_emb
    rotary.rope_type = "dynamic"
    with pytest.raises(ValueError, match="disable --persistent-workers"):
        persistent.validate_worker_models(target, draft)


def test_failed_spawn_closes_and_reaps_existing_workers(monkeypatch):
    children = []

    def launch(*_args, **_kwargs):
        if children:
            raise OSError("spawn failed")
        child = SimpleNamespace(stdin=io.StringIO(), code=None)
        child.poll = lambda: child.code

        def terminate():
            child.code = -15

        child.terminate = terminate
        child.wait = lambda **_kwargs: child.code
        children.append(child)
        return child

    monkeypatch.setattr(persistent.subprocess, "Popen", launch)
    pool = persistent.PersistentWorkers()

    def dispatch():
        with pool:
            pool.submit(0, ["python", "eval.py"], {})
            pool.submit(1, ["python", "eval.py"], {})

    with pytest.raises(OSError, match="spawn failed"):
        dispatch()
    assert children[0].stdin.closed
    assert children[0].code == -15
    assert not pool.directory.exists()


@pytest.mark.parametrize("failure", ["settings", "evaluation"])
def test_failed_job_never_publishes_completion(tmp_path, monkeypatch, failure):
    args = save_fixture(tmp_path)
    args.worker_shard_index = 0
    provenance.prepare_checkpoints(args)
    command = parallel.worker_command(
        args,
        entrypoint=Path("scripts/evaluate/leagle_offline_eval.py"),
        dataset_path=args.datasets_root,
        shard_index=0,
        num_shards=1,
        output_dir=tmp_path / "failed-job",
    )
    if failure == "settings":
        command[command.index("--spec-tokens") + 1] = "8"
    marker = tmp_path / "failed.done"
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps({"argv": command[1:], "completion": str(marker)}) + "\n"
        ),
    )
    monkeypatch.setattr(leagle, "save_provenance", lambda *_args, **_kwargs: None)

    def evaluate(*_args):
        raise RuntimeError("injected evaluation error")

    monkeypatch.setattr(leagle, "evaluate_dataset", evaluate)
    runner = SimpleNamespace(
        device=torch.device("cpu"), draft_model=SimpleNamespace(config=None)
    )
    expected = (
        (ValueError, "same model")
        if failure == "settings"
        else (RuntimeError, "injected")
    )
    with pytest.raises(expected[0], match=expected[1]):
        persistent.serve_jobs(args, runner, [])
    assert not marker.exists()
