"""Dataset identities and output isolation for the offline evaluator."""

# ruff: noqa: INP001 -- Existing evaluate test directory is not a package.

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_module():
    path = Path(__file__).parents[3] / "scripts" / "evaluate" / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_dataset_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_dataset(path, prompt):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompt": prompt}) + "\n", encoding="utf-8")


def _args(root, output):
    return SimpleNamespace(
        datasets_root=root,
        output_dir=output,
        max_samples=None,
        seed=1,
        no_progress=True,
        skip_artifacts=False,
        log_every=1,
        raw_prompt_mode="raw",
        enable_thinking="false",
        ascend_devices="0,1",
        datasets=None,
        measure_base_speedup=False,
        verifier_model="target",
        draft_model="draft",
        max_new_tokens=8,
        temperature=0.0,
        device="npu:0",
        dtype="bfloat16",
        draft_attn_impl="sdpa",
        d2t_path=None,
        t2d_path=None,
        trust_remote_code=False,
        sample_from_anchor=None,
    )


def test_nested_same_name_datasets_keep_separate_summaries_and_artifacts(tmp_path):
    module = _load_module()
    root = tmp_path / "datasets"
    _write_dataset(root / "math" / "test.jsonl", "math prompt")
    _write_dataset(root / "code" / "test.jsonl", "code prompt")
    args = _args(root, tmp_path / "out")

    def generate_one(prompt, _stop_tokens):
        token = 10 if prompt == "math prompt" else 20
        return SimpleNamespace(
            output_ids=[SimpleNamespace(tolist=lambda: [token])],
            num_input_tokens=1,
            num_output_tokens=3,
            proposal_lengths=[2],
            accepted_draft_lengths=[2],
            accept_prob_lists=[[1.0, 1.0]],
            support_accept_rate_lists=[[1.0, 1.0]],
        )

    runner = SimpleNamespace(tokenizer=object(), generate_one=generate_one)
    rows, artifacts = [], {}
    for path in module._discover_datasets(root, None):
        row, generated = module._evaluate_dataset(
            path=path,
            runner=runner,
            base_runner=None,
            args=args,
            stop_token_ids=None,
        )
        rows.append(row)
        artifacts[row["dataset"]] = generated
    module._write_outputs(args.output_dir, rows, artifacts)

    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert [row["dataset"] for row in summary] == ["code/test", "math/test"]
    assert [row["num_requests"] for row in summary] == [1, 1]
    for dataset, token in [("code/test", 20), ("math/test", 10)]:
        records = module._read_worker_artifacts(args.output_dir, dataset)
        assert records[0]["output_token_ids"] == [token]
        assert (args.output_dir / "artifacts" / f"{dataset}.jsonl").is_file()


def test_flat_and_single_file_roots_keep_existing_names(tmp_path):
    module = _load_module()
    path = tmp_path / "math500.jsonl"
    _write_dataset(path, "prompt")

    assert module._dataset_id(path, tmp_path) == "math500"
    assert module._dataset_id(path, path) == "math500"
    module._write_outputs(tmp_path / "out", [], {"math500": [{"value": 1}]})
    assert module._read_worker_artifacts(tmp_path / "out", "math500") == [{"value": 1}]
    assert (tmp_path / "out" / "artifacts" / "math500.jsonl").is_file()


def test_dataset_identity_preserves_alias_names(tmp_path, monkeypatch):
    module = _load_module()
    alias = tmp_path / "alias.jsonl"
    other_alias = tmp_path / "other_alias.jsonl"
    target = tmp_path / "real.jsonl"
    for path in [alias, other_alias, target]:
        _write_dataset(path, "prompt")
    original_resolve = Path.resolve

    def resolve_alias(path, *args, **kwargs):
        if path in {alias, other_alias}:
            return target
        return original_resolve(path, *args, **kwargs)

    # Simulate aliases without requiring Windows symlink privileges.
    monkeypatch.setattr(Path, "resolve", resolve_alias)
    assert module._dataset_id(alias, alias) == "alias"
    assert module._dataset_id(alias, tmp_path) == "alias"
    assert module._dataset_id(other_alias, tmp_path) == "other_alias"


def test_nested_dataset_can_be_selected_by_its_summary_identity(tmp_path):
    module = _load_module()
    math_path = tmp_path / "math" / "test.jsonl"
    code_path = tmp_path / "code" / "test.jsonl"
    _write_dataset(math_path, "math prompt")
    _write_dataset(code_path, "code prompt")

    assert module._discover_datasets(tmp_path, ["math/test"]) == [math_path]
    assert module._discover_datasets(tmp_path, ["test"]) == [code_path, math_path]


def test_nested_dataset_still_uses_basename_for_default_sample_cap(tmp_path):
    module = _load_module()
    root = tmp_path / "datasets"
    path = root / "nested" / "math500.jsonl"
    _write_dataset(path, "prompt")
    selected_names = []

    def select_records(records, *, dataset_name, **_kwargs):
        selected_names.append(dataset_name)
        return []

    module._select_eval_records = select_records
    row, _ = module._evaluate_dataset(
        path=path,
        runner=SimpleNamespace(tokenizer=object()),
        base_runner=None,
        args=_args(root, tmp_path / "out"),
        stop_token_ids=None,
    )
    assert selected_names == ["math500"]
    assert row["dataset"] == "nested/math500"


@pytest.mark.parametrize(
    "dataset", ["../outside", "/absolute", "C:/drive", "a\\b", "a//b"]
)
def test_dataset_output_rejects_paths_outside_identity_namespace(tmp_path, dataset):
    module = _load_module()
    with pytest.raises(ValueError, match="dataset identity"):
        module._dataset_output_path(tmp_path / "artifacts", dataset, ".jsonl")


def test_dataset_identity_rejects_file_outside_root(tmp_path):
    module = _load_module()
    root = tmp_path / "datasets"
    root.mkdir()
    outside = tmp_path / "outside.jsonl"
    _write_dataset(outside, "prompt")
    with pytest.raises(ValueError, match="outside the input root"):
        module._dataset_id(outside, root)


def test_data_parallel_parent_restores_nested_ids_from_flat_worker_outputs(
    tmp_path, monkeypatch
):
    module = _load_module()
    root = tmp_path / "datasets"
    _write_dataset(root / "math" / "test.jsonl", "math prompt")
    _write_dataset(root / "code" / "test.jsonl", "code prompt")
    args = _args(root, tmp_path / "out")
    worker_directories = []

    class Worker:
        def __init__(self, command, *, env):
            dataset_path = Path(command[command.index("--datasets-root") + 1])
            output = Path(command[command.index("--output-dir") + 1])
            shard_index = int(command[command.index("--worker-shard-index") + 1])
            assert env["ASCEND_RT_VISIBLE_DEVICES"] == str(shard_index)
            worker_directories.append(output.relative_to(args.output_dir).as_posix())
            # Worker roots are individual files, so worker artifacts use only
            # the basename even when the parent's dataset identity is nested.
            row = module._summary_row(
                dataset_path.stem,
                1,
                module.EvalStats(
                    elapsed_s=1.0,
                    total_output_tokens=1,
                    num_proposals=0,
                ),
            )
            module._write_outputs(
                output,
                [row],
                {
                    dataset_path.stem: [
                        {
                            "source_index": shard_index + 1,
                            "source": dataset_path.parent.name,
                        }
                    ]
                },
            )

        @staticmethod
        def wait():
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Worker)
    module.run_ascend_data_parallel(args)

    assert worker_directories == [
        "_shards/code/test/shard_0",
        "_shards/code/test/shard_1",
        "_shards/math/test/shard_0",
        "_shards/math/test/shard_1",
    ]
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert [row["dataset"] for row in summary] == ["code/test", "math/test"]
    assert [row["num_requests"] for row in summary] == [2, 2]
    for name in ["code", "math"]:
        assert module._read_worker_artifacts(args.output_dir, f"{name}/test") == [
            {"source_index": 1, "source": name},
            {"source_index": 2, "source": name},
        ]
