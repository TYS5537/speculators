"""Real Arrow/torch pipeline tests without a model or running vLLM."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from scripts import prepare_data as entry
from speculators.data_generation import preprocessing as generic
from speculators_dsv4 import preprocessing as prep
from tests.standalone.test_dsv4_preprocessing import TextRenderer


@pytest.fixture
def environment(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    report = {
        "model_path": str(model),
        "checkpoint_signature": "sig",
        "config": {"vocab_size": 1000, "eos_token_id": 1},
    }
    monkeypatch.setattr(prep, "inspect_checkpoint", lambda _path: report)
    monkeypatch.setattr(
        prep,
        "connect_encoder",
        lambda args, _report: prep.DSV4TrainingEncoder(
            TextRenderer(), enable_thinking=bool(args.enable_thinking)
        ),
    )
    args = SimpleNamespace(
        model=str(model),
        data=[str(tmp_path / "input.jsonl")],
        seq_length=100,
        max_samples=None,
        minimum_valid_tokens=None,
        seed=7,
        num_preprocessing_workers=1,
        trust_remote_code=False,
        assistant_pattern=None,
        enable_thinking=False,
        allow_empty_output=False,
        dsv4_source_manifest=None,
    )
    return args, report


def test_pretokenized_generic_entry_never_loads_processor(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("An encoded dataset must not load a processor or visualize text")

    monkeypatch.setattr(generic, "load_processor", forbidden)
    monkeypatch.setattr(generic, "_visualize_sample", forbidden)
    source = tmp_path / "encoded"
    Dataset.from_dict(
        {"input_ids": [[1, 2, 3]], "loss_mask": [[0, 1, 1]]}
    ).save_to_disk(str(source))
    result, processor = generic.load_and_preprocess_dataset(
        "model-with-no-chat-template",
        [str(source)],
        seq_length=2,
        build_dataset_num_proc=None,
        token_freq_path=tmp_path / "freq.pt",
    )
    assert processor is None
    assert result[0]["input_ids"].tolist() == [1, 2]
    assert result[0]["loss_mask"].tolist() == [0, 1]
    assert torch.load(tmp_path / "freq.pt", weights_only=True) == {2: 1}


def test_raw_text_to_arrow_and_frequency(environment, tmp_path):
    args, report = environment
    source = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "AA"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "B"},
        ]
    }
    Path(args.data[0]).write_text(json.dumps(source) + "\n", encoding="utf-8")
    dataset, metadata = prep.prepare_dsv4_dataset(args, tmp_path / "freq.pt")
    assert len(dataset) == 2
    assert set(dataset.column_names) == {"input_ids", "loss_mask", "seq_len"}
    assert isinstance(dataset[0]["input_ids"], torch.Tensor)
    assert torch.load(tmp_path / "freq.pt", weights_only=True) == {65: 2, 66: 1, 1: 2}
    output = tmp_path / "output"
    dataset.save_to_disk(str(output))
    (output / prep.DATA_MANIFEST).write_text(json.dumps(metadata), encoding="utf-8")
    assert len(load_from_disk(str(output))) == 2
    assert prep.validate_data_manifest(output, report)["row_count"] == 2


def test_repackage_existing_contract_without_server(environment, tmp_path, monkeypatch):
    args, report = environment
    original = tmp_path / "original"
    Dataset.from_dict(
        {"input_ids": [[0, 3, 65, 1]], "loss_mask": [[0, 0, 1, 1]]}
    ).save_to_disk(str(original))
    metadata = {
        **prep.data_identity(report),
        "encoding": prep.ENCODING,
        "enable_thinking": False,
        "mask_policy": "current-assistant-continuation-including-eos",
    }
    (original / prep.DATA_MANIFEST).write_text(json.dumps(metadata), encoding="utf-8")
    args.data = [str(original)]
    monkeypatch.setattr(
        prep, "connect_encoder", lambda *args: pytest.fail("must not contact server")
    )
    dataset, new_metadata = prep.prepare_dsv4_dataset(args, tmp_path / "freq.pt")
    assert dataset[0]["input_ids"].tolist() == [0, 3, 65, 1]
    assert new_metadata["pretokenized_source"] == metadata


def test_foreign_encoded_data_has_no_implicit_binding(environment, tmp_path):
    args, _report = environment
    original = tmp_path / "original"
    Dataset.from_dict(
        {"input_ids": [[0, 3, 65, 1]], "loss_mask": [[0, 0, 1, 1]]}
    ).save_to_disk(str(original))
    args.data = [str(original)]
    with pytest.raises(ValueError, match="Missing DSV4"):
        prep.prepare_dsv4_dataset(args, tmp_path / "freq.pt")


def test_json_directory_ignores_contract_sidecars(environment, tmp_path):
    args, report = environment
    original = tmp_path / "original-json"
    original.mkdir()
    (original / "input.jsonl").write_text(
        json.dumps({"input_ids": [0, 3, 65, 1], "loss_mask": [0, 0, 1, 1]}),
        encoding="utf-8",
    )
    metadata = {
        **prep.data_identity(report),
        "encoding": prep.ENCODING,
        "enable_thinking": False,
        "mask_policy": "current-assistant-continuation-including-eos",
    }
    (original / prep.DATA_MANIFEST).write_text(json.dumps(metadata), encoding="utf-8")
    args.data = [str(original)]
    dataset, _metadata = prep.prepare_dsv4_dataset(args, tmp_path / "freq.pt")
    assert len(dataset) == 1


def test_truncation_drops_no_signal_rows(environment, tmp_path):
    args, _report = environment
    args.seq_length = 2
    args.allow_empty_output = True
    Path(args.data[0]).write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "long question"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset, metadata = prep.prepare_dsv4_dataset(args, tmp_path / "freq.pt")
    assert len(dataset) == 0
    assert metadata["row_count"] == 0
    assert torch.load(tmp_path / "freq.pt", weights_only=True) == {}


@pytest.mark.parametrize(
    "tokenizer_class",
    ["CachedDSV4TokenizersBackend", "TokenizerPoolCachedDSV4TokenizersBackend"],
)
def test_connect_checks_the_official_wrapped_tokenizer_class(
    tmp_path, monkeypatch, tokenizer_class
):
    local = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    local.save(str(tmp_path / "tokenizer.json"))
    report = {
        "model_path": str(tmp_path),
        "checkpoint_signature": "sig",
        "config": {"vocab_size": 10, "eos_token_id": 1},
    }
    (tmp_path / prep.MANIFEST).write_text(json.dumps(report), encoding="utf-8")
    args = SimpleNamespace(
        dsv4_tokenizer_endpoint="http://127.0.0.1:1234",
        dsv4_served_model_name="target",
        dsv4_hs_manifest=str(tmp_path),
        dsv4_tokenizer_timeout=10,
        enable_thinking=False,
    )

    class Client:
        def __init__(self, *args):
            pass

        def request(self, path):
            return {
                "/version": {"version": "0.26.0+ascend"},
                "/v1/models": {"data": [{"id": "target", "root": str(tmp_path)}]},
                "/tokenizer_info": {"tokenizer_class": tokenizer_class},
            }[path]

        def post(self, path, *, cast_to, body):
            assert path == "/tokenize"
            assert cast_to is dict
            ids = local.encode(body["prompt"], add_special_tokens=False).ids
            return {"tokens": ids, "count": len(ids)}

    monkeypatch.setattr(prep, "TokenizationClient", Client)
    assert isinstance(prep.connect_encoder(args, report), prep.DSV4TrainingEncoder)


def test_entry_rejects_overwrite_of_input_before_removing_it(tmp_path, monkeypatch):
    input_path = tmp_path / "data-00000-of-00001.arrow"
    input_path.write_bytes(b"keep")
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--model",
            "target",
            "--data",
            str(tmp_path),
            "--output",
            str(tmp_path),
            "--overwrite",
        ],
    )
    with pytest.raises(ValueError, match="contains an input"):
        entry.main()
    assert input_path.read_bytes() == b"keep"


def test_entry_writes_dsv4_manifest(environment, tmp_path, monkeypatch):
    args, _report = environment
    Path(args.data[0]).write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "entry-output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_data.py",
            "--dsv4",
            "--model",
            args.model,
            "--data",
            args.data[0],
            "--output",
            str(output),
            "--num-preprocessing-workers",
            "1",
            "--disable-thinking",
        ],
    )
    entry.main()
    assert (
        json.loads((output / prep.DATA_MANIFEST).read_text(encoding="utf-8"))[
            "row_count"
        ]
        == 1
    )
    assert load_from_disk(str(output))[0]["loss_mask"][-1] == 1
    assert torch.load(output / "token_freq.pt", weights_only=True) == {65: 1, 1: 1}
