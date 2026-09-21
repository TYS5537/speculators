import logging
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def check_hidden_states(data: dict, tokens: list[int]):
    if not {"token_ids", "hidden_states"}.issubset(data):
        raise ValueError(
            "Hidden-state payload must contain token_ids and hidden_states"
        )
    if data["token_ids"].ndim != 1:
        raise ValueError("Hidden-state token IDs must be one-dimensional")
    t_ids = data["token_ids"].tolist()
    if t_ids != tokens:
        raise ValueError(f"Token ids don't match expected token ids {tokens}")

    hs = data["hidden_states"]
    expected_ndim = 3  # [tokens, auxiliary slots + teacher, hidden width]
    # MTP may export only the teacher slot; auxiliary slots are model-dependent.
    if hs.ndim != expected_ndim or hs.shape[1] < 1 or hs.shape[2] < 1:
        raise ValueError(
            "Hidden states must have shape [tokens, auxiliary+teacher, width]"
        )
    if not hs.is_floating_point() or not hs.isfinite().all():
        raise ValueError("Hidden states must be floating-point with no NaN/Inf values")
    if len(tokens) != hs.shape[0]:
        raise ValueError(
            f"Sequence length of hidden states {hs.shape[0]}"
            f" doesn't match num tokens {len(tokens)}"
        )


def align_hidden_states(data: dict, tokens: list[int], *, allow_prefix: bool = False):
    """Select an exact causal prefix from an otherwise complete HS response.

    Multimodal requests must retain the complete messages/images at the server.
    Only their returned states may be shortened, after validating token identity
    and the full response shape. Text requests keep exact-length validation.
    """
    if not allow_prefix:
        check_hidden_states(data, tokens)
        return data
    if "token_ids" not in data or data["token_ids"].ndim != 1:
        raise ValueError("Hidden-state token IDs must be one-dimensional")
    actual = data["token_ids"].tolist()
    if len(actual) < len(tokens) or actual[: len(tokens)] != tokens:
        raise ValueError("Hidden-state token IDs do not match the training prefix")
    check_hidden_states(data, actual)
    return data | {
        "token_ids": data["token_ids"][: len(tokens)],
        "hidden_states": data["hidden_states"][: len(tokens)],
    }


def validate_existing_hidden_states(output_path: Path, dataset, indices: list[int]):
    """Keep valid caches; preserve invalid files under unique quarantine names.

    The caller supplies only this rank's requested rows, so other ranks and files
    beyond max_samples are never renamed. Quarantined files are ignored by cache
    discovery and can be inspected or recovered by the user.
    """
    from safetensors import SafetensorError  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    valid = []
    for index in indices:
        path = output_path / f"hs_{index}.safetensors"
        item = dataset[index]
        tokens = item["input_ids"].tolist()
        allow_prefix = any(
            isinstance(message.get("content"), list)
            for message in item.get("messages", [])
        )
        try:
            align_hidden_states(load_file(path), tokens, allow_prefix=allow_prefix)
        except (KeyError, ValueError, RuntimeError, SafetensorError) as exc:
            quarantined = path.with_name(f"{path.name}.invalid-{uuid4().hex}")
            path.rename(quarantined)
            logger.warning(
                "Invalid HS cache for row %d preserved at %s; regenerating: %s",
                index,
                quarantined,
                exc,
            )
        else:
            valid.append(index)
    return valid


def get_existing_hidden_state_indices(output_path: Path) -> list[int]:
    """Find existing `hs_i.safetensors` files (where i is the file index)"""

    existing_file_indices_set: set[int] = set()

    if not output_path.exists():
        return []

    for file_path in output_path.iterdir():
        if file_path.name.startswith("hs_") and file_path.name.endswith(".safetensors"):
            index_str = file_path.stem[3:]  # Remove "hs_" prefix
            try:
                file_index = int(index_str)
                existing_file_indices_set.add(file_index)
            except ValueError:
                continue

    return sorted(existing_file_indices_set)


def get_indices_to_process(
    num_samples: int,
    max_samples: int | None,
    existing: list[int],
    world_size: int,
    rank: int,
) -> list[int]:
    """Determines which indices should be processed. If max_samples is None
    returns all dataset indices not in existing. Otherwise gets the first
    `max_samples - len(existing)` samples not already in existing.

    Args:
        num_samples: Total size of preprocessed dataset
        max_samples: (Optional) limit for number of samples to process
        existing: list of ids that have already been processed
        world_size: Number of nodes to generate on
        rank: The rank of the local node

    Returns:
        list of dataset indices to process
    """

    target = min(max_samples, num_samples) if max_samples is not None else num_samples

    if target <= 0:
        return []

    chunk_size = target // world_size
    remainder = target % world_size
    # Distribute remainder across the first `remainder` ranks so chunks differ
    # by at most 1.
    start = rank * chunk_size + min(rank, remainder)
    end = start + chunk_size + (1 if rank < remainder else 0)

    existing_s = set(existing)
    to_process = [i for i in range(start, end) if i not in existing_s]

    if not to_process:
        logger.info("All samples for this rank already processed!")
        return []

    if len(existing_s & set(range(start, end))) > 0:
        logger.info(
            f"Found {len(existing_s & set(range(start, end)))} existing samples"
            f" for rank {rank}."
        )

    return to_process
