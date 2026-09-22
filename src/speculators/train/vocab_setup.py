"""Resolve, cache, and distribute vocabulary mappings for draft training."""

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np
import torch

from speculators.models.utils import get_verifier_config
from speculators.train.distributed import get_rank, is_distributed
from speculators.train.vocab_mapping import (
    build_vocab_mappings_from_distribution,
    get_target_vocab_size,
)

_LOGGER = logging.getLogger(__name__)


def _load_mappings(
    d2t_path,
    t2d_path,
    expected_draft_vocab_size: int | None,
    *,
    logger: logging.Logger | None = None,
):
    logger = _LOGGER if logger is None else logger
    logger.info(f"Loading vocab mappings from '{d2t_path}' and '{t2d_path}'")
    # Load d2t and t2d tensors if provided
    d2t = torch.from_numpy(np.load(d2t_path))
    t2d = torch.from_numpy(np.load(t2d_path))
    draft_vocab_size = d2t.shape[0]
    if expected_draft_vocab_size and expected_draft_vocab_size != draft_vocab_size:
        raise ValueError(
            f"Explicit vocab mapping (t2d & d2t) files were provided, but don't"
            f"match the provided --draft-vocab-size {draft_vocab_size}."
            f"d2t.shape={d2t.shape}, dim 0 should match provided value."
        )
    return d2t, t2d, draft_vocab_size


def _save_vocab_mapping_atomically(path: Path, values: np.ndarray) -> None:
    """Publish a complete numpy file, never a visible partially written cache."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            np.save(stream, values)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_vocab_mappings_local(
    args: argparse.Namespace, *, logger: logging.Logger | None = None
):
    logger = _LOGGER if logger is None else logger
    if args.d2t_path or args.t2d_path:
        if not (args.d2t_path and args.t2d_path):
            raise ValueError(
                "Both t2d and d2t must be provided together, or both must be omitted. "
                f"Got t2d={'provided' if args.t2d_path is not None else 'not provided'}"
                f"d2t={'provided' if args.d2t_path is not None else 'not provided'}"
            )

        return _load_mappings(
            args.d2t_path, args.t2d_path, args.draft_vocab_size, logger=logger
        )

    data_path = Path(args.data_path)
    default_t2d_path = data_path / "t2d.npy"
    default_d2t_path = data_path / "d2t.npy"

    if default_t2d_path.exists() and default_d2t_path.exists():
        return _load_mappings(
            default_d2t_path, default_t2d_path, args.draft_vocab_size, logger=logger
        )

    token_freq_path = args.token_freq_path or data_path / "token_freq.pt"
    token_freq_path = Path(token_freq_path)
    if token_freq_path.exists() and args.draft_vocab_size is not None:
        logger.info("No vocab mappings provided. Regenerating from token frequencies")
        token_freq_dict = torch.load(token_freq_path, weights_only=True)

        target_vocab_size = get_target_vocab_size(None, args.verifier_name_or_path)

        d2t, t2d = build_vocab_mappings_from_distribution(
            token_freq_dict=token_freq_dict,
            draft_vocab_size=args.draft_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        draft_vocab_size = d2t.shape[0]
        if args.draft_vocab_size and args.draft_vocab_size != draft_vocab_size:
            raise ValueError(
                f"Explicit vocab mapping (t2d & d2t) files were provided, but don't"
                f"match the provided --draft-vocab-size {draft_vocab_size}."
                f"d2t.shape={d2t.shape}, dim 0 should match provided value."
            )

        logger.info(f"Caching vocab mapping files to '{data_path}'")
        _save_vocab_mapping_atomically(data_path / "d2t.npy", d2t.cpu().numpy())
        _save_vocab_mapping_atomically(data_path / "t2d.npy", t2d.cpu().numpy())

        return d2t, t2d, draft_vocab_size

    logger.warning(
        "No vocab mappings found, and can't generate new ones because either "
        f"token_freq_path='{token_freq_path}' doesn't exist or --draft-vocab-size is "
        "None. Using full verifier vocab"
    )
    # When vocab mapping is not provided, use the full verifier vocab
    verifier_config = get_verifier_config(args.verifier_name_or_path)
    return None, None, verifier_config.vocab_size


def parse_vocab_mappings(
    args: argparse.Namespace, *, logger: logging.Logger | None = None
):
    """Resolve mappings once, then share the same result or failure with all ranks."""
    logger = _LOGGER if logger is None else logger
    if not is_distributed():
        return _parse_vocab_mappings_local(args, logger=logger)

    # All ranks must enter this collective, including when rank zero cannot read
    # or create the mappings. A barrier after unguarded I/O would leave peers
    # waiting forever on a rank-zero failure. CPU mappings are small startup data;
    # broadcasting them also avoids requiring a shared cache filesystem.
    payload = [{}]
    if get_rank() == 0:
        try:
            payload[0] = {"mappings": _parse_vocab_mappings_local(args, logger=logger)}
        except Exception as exc:  # noqa: BLE001 -- report failure to every rank.
            payload[0] = {"error": f"{type(exc).__name__}: {exc}"}
    torch.distributed.broadcast_object_list(payload, src=0)
    result = payload[0]
    if "error" in result:
        raise ValueError(
            "Vocabulary mapping setup failed on rank 0: " + result["error"]
        )
    return result["mappings"]
