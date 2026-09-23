"""Vocabulary startup entrypoints and source precedence across the module split."""

import argparse
import inspect
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from scripts import train
from speculators.train import vocab_setup


def _args(directory, **overrides):
    values = {
        "data_path": str(directory),
        "d2t_path": None,
        "t2d_path": None,
        "token_freq_path": None,
        "draft_vocab_size": 3,
        "verifier_name_or_path": "verifier",
    }
    return argparse.Namespace(**(values | overrides))


@pytest.fixture(autouse=True)
def local_rank(monkeypatch):
    monkeypatch.setattr(vocab_setup, "is_distributed", lambda: False)


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("_load_mappings", ("d2t.npy", "t2d.npy", 3)),
        ("_save_vocab_mapping_atomically", (object(), object())),
        ("_parse_vocab_mappings_local", (_args("data"),)),
        ("parse_vocab_mappings", (_args("data"),)),
    ],
)
def test_script_wrappers_preserve_arguments_and_dynamic_logger(
    monkeypatch, name, arguments
):
    signature = inspect.signature(getattr(vocab_setup, name))
    wrapper = getattr(train, name)
    wrapper_signature = inspect.signature(wrapper)
    assert list(wrapper_signature.parameters) == [
        key for key in signature.parameters if key != "logger"
    ]
    implementation = Mock(
        return_value=None if name == "_save_vocab_mapping_atomically" else object()
    )
    monkeypatch.setattr(vocab_setup, name, implementation)

    for caller_logger in (Mock(), Mock()):
        monkeypatch.setattr(train, "logger", caller_logger)
        assert wrapper(*arguments) is implementation.return_value
        actual = signature.bind(
            *implementation.call_args.args, **implementation.call_args.kwargs
        )
        expected = wrapper_signature.bind(*arguments).arguments
        if name != "_save_vocab_mapping_atomically":
            expected["logger"] = caller_logger
        assert actual.arguments == expected
    assert implementation.call_count == 2


def _write_pair(directory, *, prefix=""):
    d2t = np.array([0, 1, 2], dtype=np.int32)
    t2d = np.array([True, False, True, False, True, False, False, False])
    d2t_path, t2d_path = directory / f"{prefix}d2t.npy", directory / f"{prefix}t2d.npy"
    np.save(d2t_path, d2t)
    np.save(t2d_path, t2d)
    return d2t_path, t2d_path, d2t, t2d


@pytest.mark.parametrize("expected_size", [0, 3])
def test_explicit_pair_wins_over_cache_and_preserves_dtype(
    tmp_path, monkeypatch, expected_size
):
    d2t_path, t2d_path, d2t, t2d = _write_pair(tmp_path, prefix="explicit-")
    (tmp_path / "d2t.npy").write_bytes(b"invalid cache")
    (tmp_path / "t2d.npy").write_bytes(b"invalid cache")
    forbidden = Mock(side_effect=AssertionError("Explicit pair must bypass generation"))
    monkeypatch.setattr(
        vocab_setup, "build_vocab_mappings_from_distribution", forbidden
    )
    monkeypatch.setattr(vocab_setup, "get_verifier_config", forbidden)
    caller_logger = Mock()

    actual_d2t, actual_t2d, size = vocab_setup.parse_vocab_mappings(
        _args(
            tmp_path,
            d2t_path=str(d2t_path),
            t2d_path=str(t2d_path),
            draft_vocab_size=expected_size,
        ),
        logger=caller_logger,
    )

    assert size == 3  # An expected size of zero retains the existing truthy guard.
    assert actual_d2t.dtype == torch.int32
    assert actual_t2d.dtype == torch.bool
    np.testing.assert_array_equal(actual_d2t.numpy(), d2t)
    np.testing.assert_array_equal(actual_t2d.numpy(), t2d)
    caller_logger.info.assert_called_once()
    assert str(d2t_path) in caller_logger.info.call_args.args[0]
    forbidden.assert_not_called()


def test_cached_pair_precedes_requested_frequency_file(tmp_path, monkeypatch):
    _, _, d2t, t2d = _write_pair(tmp_path)
    frequency_path = tmp_path / "other-frequencies.pt"
    frequency_path.write_bytes(b"must not be read")
    read_frequency = Mock(side_effect=AssertionError("Cached pair has precedence"))
    monkeypatch.setattr(torch, "load", read_frequency)

    actual_d2t, actual_t2d, size = vocab_setup.parse_vocab_mappings(
        _args(tmp_path, token_freq_path=str(frequency_path))
    )

    assert size == 3
    np.testing.assert_array_equal(actual_d2t.numpy(), d2t)
    np.testing.assert_array_equal(actual_t2d.numpy(), t2d)
    read_frequency.assert_not_called()


def test_corrupt_cached_pair_errors_without_regeneration(tmp_path, monkeypatch):
    _write_pair(tmp_path)
    (tmp_path / "d2t.npy").write_bytes(b"corrupt")
    torch.save({0: 4, 2: 3, 4: 2}, tmp_path / "token_freq.pt")
    generate = Mock(
        side_effect=AssertionError("Corrupt cache must not silently rebuild")
    )
    monkeypatch.setattr(vocab_setup, "build_vocab_mappings_from_distribution", generate)

    with pytest.raises(ValueError):
        vocab_setup.parse_vocab_mappings(_args(tmp_path))

    generate.assert_not_called()
    assert (tmp_path / "d2t.npy").read_bytes() == b"corrupt"


def test_requested_frequency_file_precedes_implicit_frequency_file(
    tmp_path, monkeypatch
):
    torch.save({0: 9, 2: 7, 4: 5}, tmp_path / "token_freq.pt")
    frequency_path = tmp_path / "requested.pt"
    torch.save({1: 9, 3: 7, 5: 5}, frequency_path)
    target_size = Mock(return_value=8)
    monkeypatch.setattr(vocab_setup, "get_target_vocab_size", target_size)

    d2t, t2d, size = vocab_setup.parse_vocab_mappings(
        _args(tmp_path, token_freq_path=str(frequency_path))
    )

    assert size == 3
    assert d2t.tolist() == [1, 2, 3]
    assert t2d.nonzero().flatten().tolist() == [1, 3, 5]
    target_size.assert_called_once_with(None, "verifier")
    np.testing.assert_array_equal(np.load(tmp_path / "d2t.npy"), d2t.numpy())
    np.testing.assert_array_equal(np.load(tmp_path / "t2d.npy"), t2d.numpy())


@pytest.mark.parametrize("draft_size", [None, 0])
def test_none_skips_generation_but_zero_is_still_an_explicit_generation_size(
    tmp_path, monkeypatch, caplog, draft_size
):
    frequency = {0: 5}
    torch.save(frequency, tmp_path / "token_freq.pt")
    empty_d2t, empty_t2d = (
        torch.empty(0, dtype=torch.long),
        torch.zeros(8, dtype=torch.bool),
    )
    generate = Mock(return_value=(empty_d2t, empty_t2d))
    verifier = Mock(return_value=SimpleNamespace(vocab_size=8))
    monkeypatch.setattr(vocab_setup, "build_vocab_mappings_from_distribution", generate)
    monkeypatch.setattr(vocab_setup, "get_target_vocab_size", Mock(return_value=8))
    monkeypatch.setattr(vocab_setup, "get_verifier_config", verifier)

    d2t, t2d, size = vocab_setup.parse_vocab_mappings(
        _args(tmp_path, draft_vocab_size=draft_size)
    )

    if draft_size is None:
        assert (d2t, t2d, size) == (None, None, 8)
        verifier.assert_called_once_with("verifier")
        generate.assert_not_called()
        assert not (tmp_path / "d2t.npy").exists()
        assert len(caplog.records) == 1
        assert caplog.records[0].name == vocab_setup.__name__
    else:
        assert d2t is empty_d2t
        assert t2d is empty_t2d
        assert size == 0
        generate.assert_called_once_with(
            token_freq_dict=frequency, draft_vocab_size=0, target_vocab_size=8
        )
        verifier.assert_not_called()


def test_size_mismatch_does_not_fall_back_to_generation(tmp_path, monkeypatch):
    _write_pair(tmp_path)
    generate = Mock()
    monkeypatch.setattr(vocab_setup, "build_vocab_mappings_from_distribution", generate)

    with pytest.raises(ValueError, match="draft-vocab-size"):
        vocab_setup.parse_vocab_mappings(_args(tmp_path, draft_vocab_size=4))

    generate.assert_not_called()


def test_broadcast_failure_is_not_relabelled_as_rank_zero_mapping_failure(monkeypatch):
    monkeypatch.setattr(vocab_setup, "is_distributed", lambda: True)
    monkeypatch.setattr(vocab_setup, "get_rank", lambda: 0)
    monkeypatch.setattr(
        vocab_setup, "_parse_vocab_mappings_local", Mock(return_value=(None, None, 8))
    )
    failure = RuntimeError("collective unavailable")
    broadcast = Mock(side_effect=failure)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)

    with pytest.raises(RuntimeError, match="collective unavailable") as caught:
        vocab_setup.parse_vocab_mappings(_args("not-read"))

    assert caught.value is failure
    broadcast.assert_called_once_with([{"mappings": (None, None, 8)}], src=0)
