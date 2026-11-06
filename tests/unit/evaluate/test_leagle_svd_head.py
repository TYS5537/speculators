"""SVD inference from dense checkpoints, worker flags and provenance."""

import json
from pathlib import Path

import pytest
import torch

from speculators import SpeculatorModel
from speculators_eval import leagle, parallel
from speculators_eval.leagle_decoding import check_greedy_output, generate_speculative

from .test_eagle3_offline_eval import make_eagle
from .test_leagle_offline_eval import make_real_models, save_fixture


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(42)
    torch.set_num_threads(1)


@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sparse", [False, True])
def test_svd_head_cached_and_rebuilt_generation(temperature, reduced, dtype, sparse):
    target, draft = make_real_models(layers=2, loops=3, reduced=reduced)
    target, draft = target.to(dtype), draft.to(dtype)
    draft.set_svd_candidate_head(4, top_k=3)
    draft.set_inference_optimizations(sparse_candidates=sparse)
    inputs = torch.tensor([[5, 9, 2]])
    responses = []
    for reuse in (False, True):
        torch.manual_seed(53)
        response = generate_speculative(
            target,
            draft,
            inputs,
            max_new_tokens=19,
            temperature=temperature,
            draft_kv_reuse=reuse,
        )
        responses.append(response)
        if temperature == 0:
            check_greedy_output(target, inputs, response, 19, [])
        assert response.num_output_tokens == 19
    torch.testing.assert_close(responses[0].output_ids, responses[1].output_ids)
    assert responses[0].accepted_draft_lengths == responses[1].accepted_draft_lengths


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_dense_checkpoint_svd_eval_does_not_rewrite_or_resave_runtime_head(
    tmp_path, temperature, caplog
):
    args = save_fixture(tmp_path)
    checkpoint = Path(args.draft_model)
    before = {p.name: p.read_bytes() for p in checkpoint.iterdir() if p.is_file()}
    args.candidate_svd_rank, args.candidate_top_k = 5, 2
    args.temperature, args.check_greedy = temperature, temperature == 0
    entrypoint = Path("scripts/evaluate/leagle_offline_eval.py")
    worker = parallel.worker_command(
        args,
        entrypoint=entrypoint,
        dataset_path=args.datasets_root,
        shard_index=1,
        num_shards=2,
        output_dir=tmp_path / "worker",
    )
    child = leagle.build_parser().parse_args(worker[2:])
    assert child.candidate_svd_rank == 5
    assert child.candidate_top_k == 2
    with caplog.at_level("INFO", logger="speculators_eval"):
        leagle.run(args, entrypoint=entrypoint)
    assert "SVD initialization finished" in caplog.text
    assert before == {
        p.name: p.read_bytes() for p in checkpoint.iterdir() if p.is_file()
    }
    metadata = json.loads((args.output_dir / "eval_config.json").read_text())
    assert metadata["candidate_head"]["source"] == "svd"
    assert metadata["candidate_head"]["rank"] == 5
    assert metadata["candidate_head"]["top_k"] == 2
    assert metadata["candidate_head"]["svd_method"] == "cpu-fp64-gram-eigh"
    assert not metadata["draft_config"]["candidate_head"]
    assert "candidate_svd_rank" not in metadata["draft_config"]
    _, draft, _ = leagle.load_models(args)
    assert draft.candidate_head is None
    draft.save_pretrained(tmp_path / "resaved")
    restored = SpeculatorModel.from_pretrained(tmp_path / "resaved")
    assert not restored.candidate_head_enabled
    assert restored.svd_candidate_head is None
    assert set(restored.state_dict()) == set(draft.state_dict())
    configs = [model.config.to_dict() for model in (restored, draft)]
    for config in configs:
        # HF loading changes the source path and normalizes the dtype object.
        config.pop("_name_or_path", None)
        config["dtype"] = str(config.get("dtype")).removeprefix("torch.")
    assert configs[0] == configs[1]
    for name, weight in restored.state_dict().items():
        torch.testing.assert_close(weight, draft.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("rank", [0, -1, 17])
def test_invalid_svd_rank_fails_before_model_load(tmp_path, rank):
    args = save_fixture(tmp_path)
    args.candidate_svd_rank, args.candidate_top_k = rank, 2
    with pytest.raises(ValueError, match="SVD rank|candidate-svd-rank"):
        leagle.load_models(args)


def test_svd_switch_conflicts_and_eagle_rejected(tmp_path):
    args = save_fixture(tmp_path)
    args.candidate_svd_rank, args.candidate_top_k = 4, 2
    args.candidate_head = False
    with pytest.raises(ValueError, match="no-candidate-head"):
        leagle.validate_args(args)
    args.candidate_head = None
    args.candidate_rerank = False
    with pytest.raises(ValueError, match="requires candidate reranking"):
        leagle.validate_args(args)
    args.candidate_rerank = None
    _, eagle = make_eagle()
    eagle.save_pretrained(args.draft_model)
    with pytest.raises(ValueError, match="only for LEAGLE"):
        leagle.load_models(args)


def test_svd_defaults_are_runtime_values_not_checkpoint_settings(tmp_path):
    args = save_fixture(tmp_path)
    config = json.loads((Path(args.draft_model) / "config.json").read_text())
    assert args.candidate_svd_rank is None
    assert not leagle.candidate_head_settings(args, config)["enabled"]
    args.candidate_svd_rank = 7
    config["draft_vocab_size"] = 200
    config["candidate_head_rank"] = 4
    config["candidate_head_top_k"] = 3
    config["candidate_rerank"] = False
    settings = leagle.candidate_head_settings(args, config)
    assert settings["rank"] == 7
    assert settings["top_k"] == 128
    assert settings["rerank"]
    assert settings["enabled"]
