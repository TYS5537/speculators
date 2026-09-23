import json
import logging
import math
import os
import random
import warnings
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Literal, cast

import openai
import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from hs_connectors import FileTransfer, HiddenStatesTransfer
from speculators.data_generation.offline import align_hidden_states, check_hidden_states
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ClientItem,
    InvalidResponseError,
    generate_hidden_states,
)
from speculators.train.noise_transforms import TransformTensors
from speculators.train.recovery import (
    RECOVERY_METADATA_KEY,
    GenerationRecoveryGuard,
    RecoveryMetadata,
    SampleUnavailable,
)

logger = logging.getLogger("speculators")

BatchType = dict[str, Any]


def list_files(path):
    datapath = []
    for root, _directories, files in os.walk(path):
        for file in files:
            if not file.endswith("pt"):
                continue
            file_path = Path(root) / file
            datapath.append(file_path)

    return datapath


def split_files(datapath: str, ratio: float = 0.9, seed: int = 0):
    """Given a datapath, split the files into a training and validation set
    ratio is the proportion of files to put in the training set
    1 - ratio is the proportion of files to put in the validation set
    """
    random.seed(seed)
    file_list = list_files(datapath)
    random.shuffle(file_list)
    num_files = len(file_list)
    num_train_files = int(num_files * ratio)
    train_files = file_list[:num_train_files]
    val_files = file_list[num_train_files:]
    return train_files, val_files


# Data standardization functions
StandardizeFnSig = Callable[[dict[str, Any]], dict[str, Any]]


def create_empty_sample(
    hidden_size: int, num_target_layers: int = 3, dtype: torch.dtype = torch.bfloat16
):
    # data structure: {
    #     "hidden_states": [seq_len, num_target_layers * hidden_size],
    #     "input_ids": [seq_len],
    #     "verifier_last_hidden_states": [seq_len, hidden_size],
    #     "loss_mask": [seq_len],
    #     "lengths": [1],
    #     "position_ids": [seq_len],
    # }
    # Default dtype is bfloat16 to match the hidden_states dtype used downstream.
    # When this fallback is used (e.g. vLLM hidden-state extraction times out and
    # we substitute an empty sample), the implicit float32 placeholders crashed
    # bf16 EAGLE-3 layers (fc, verifier_lm_head) with a dtype mismatch.

    return {
        "hidden_states": torch.empty(0, num_target_layers * hidden_size, dtype=dtype),
        "input_ids": torch.empty(0, dtype=torch.long),
        "verifier_last_hidden_states": torch.empty(0, hidden_size, dtype=dtype),
        "loss_mask": torch.empty(0, dtype=torch.bool),
        "lengths": torch.tensor([0], dtype=torch.long),
        "position_ids": torch.arange(0, dtype=torch.long),
    }


def standardize_data_v1(data: dict[str, Any]) -> dict[str, Any]:
    # v1 data format:
    # {
    #  "input_ids": [seq_len],
    #  "loss_mask": [seq_len],
    #  "hidden_states": [
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    [seq_len, hidden_size],
    #    ...
    #  ],
    # }

    return {
        "hidden_states": torch.cat(data["hidden_states"][:-1], dim=-1),
        "input_ids": data["input_ids"],
        "verifier_last_hidden_states": data["hidden_states"][-1],
        "loss_mask": data["loss_mask"],
    }


def _has_multimodal_content(messages: list[dict]) -> bool:
    """True when any turn carries non-text content (images, video, audio).

    Text-only turns store ``content`` as a plain string.  Multimodal turns
    (produced by ``_adapt_conv_for_vllm``) store it as a list of typed parts,
    e.g. ``[{"type": "text", ...}, {"type": "image_url", ...}]``.
    """
    return any(isinstance(m.get("content"), list) for m in messages)


def build_client_item(
    dataset_item: dict, *, legacy_final_message: bool = False
) -> ClientItem:
    """Build a request payload for vLLM hidden-state extraction.

    When ``messages`` is included, ``generate_hidden_states`` uses the Chat
    Completions API and vLLM **re-tokenizes from the raw messages**, ignoring
    ``input_ids``.  This is required for multimodal inputs (the Completions
    API cannot carry image/video/audio references), but harmful for text-only
    data: preprocessing truncates ``input_ids`` to ``seq_length``, yet the
    ``messages`` column stores the original un-truncated conversation.
    Re-tokenizing those messages produces a longer sequence that can exceed
    ``max_model_len``.

    We therefore only forward ``messages`` when the conversation actually
    contains multimodal content.  Text-only conversations always go through
    the Completions API with the pre-truncated ``input_ids``.

    This matters for models like Qwen3.5-0.8B whose ``AutoProcessor`` returns
    a ``ProcessorMixin`` (``Qwen3VLProcessor``), causing preprocessing to
    populate the ``messages`` column even for purely text-only datasets.
    Text-only EAGLE-3 models (e.g. Llama) use a plain tokenizer, so
    ``messages`` is never created and this guard is a no-op.
    """
    out_dict: dict = {"input_ids": dataset_item["input_ids"].tolist()}

    if "messages" in dataset_item and _has_multimodal_content(dataset_item["messages"]):
        out_dict["messages"] = dataset_item["messages"]
        if "continue_final_message" in dataset_item:
            out_dict["continue_final_message"] = bool(
                dataset_item["continue_final_message"]
            )
        elif legacy_final_message:
            out_dict["continue_final_message"] = False

    return cast("ClientItem", out_dict)


class BaseDataset(Dataset):
    def __init__(
        self,
        max_len: int,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
    ):
        self.max_len = max_len
        self.transform = transform
        self.hidden_states_dtype = hidden_states_dtype
        self.approx_lengths = self._compute_approx_lengths()

    def _compute_approx_lengths(self):
        raise NotImplementedError

    def _get_raw_data(self, index):
        raise NotImplementedError

    def __getitem__(self, index) -> BatchType | None:
        data = self._get_raw_data(index)

        if data is None or isinstance(data, SampleUnavailable):
            return data

        # data structure: {
        #  "hidden_states": [seq_len, 3 * hidden_size],
        #  "input_ids": [seq_len],
        #  "verifier_last_hidden_states": [seq_len, hidden_size],
        #  "loss_mask": [seq_len],
        # }

        # Add lengths tensor
        seq_len = data["input_ids"].shape[0]
        data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
        # shape: [1]

        data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
        # shape: [seq_len]

        # data structure: {
        #     "hidden_states": [seq_len, 3 * hidden_size],
        #     "input_ids": [seq_len],
        #     "verifier_last_hidden_states": [seq_len, hidden_size],
        #     "loss_mask": [seq_len],
        #     "lengths": [1],
        #     "position_ids": [seq_len],
        # }

        # Apply transform
        if self.transform:
            data = self.transform(data)

        return data


class ArrowDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | PathLike,
        transfer: HiddenStatesTransfer | None = None,
        vllm_endpoint: str = "http://localhost:8000/v1",
        on_missing: Literal["generate", "skip", "warn", "raise"] = "generate",
        on_generate: Literal["cache", "delete"] = "delete",
        split_ratio: float = 1.0,
        transform: TransformTensors | None = None,
        hidden_states_dtype=torch.bfloat16,
        model: str | None = None,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        pretokenized_text_only: bool = False,
        *,
        split: Literal["train", "validation", "val"] | None = None,
        train_ratio: float | None = None,
        generation_validation_retries: int | None = None,
        max_consecutive_generation_failures: int = 20,
    ):
        if train_ratio is not None:
            if not 0.0 < train_ratio <= 1.0:
                raise ValueError(
                    f"train_ratio must be in (0.0, 1.0], got {train_ratio}"
                )
            if split_ratio != 1.0:
                raise ValueError("train_ratio and split_ratio cannot both be supplied")
            split_ratio = train_ratio
            split = split or "train"
        if split == "val":
            split = "validation"
        self.generation_recovery = (
            GenerationRecoveryGuard(
                retries=generation_validation_retries,
                max_consecutive_failures=max_consecutive_generation_failures,
            )
            if generation_validation_retries is not None
            else None
        )
        self.pretokenized_text_only = pretokenized_text_only
        if pretokenized_text_only and max_len < 1:
            raise ValueError("External Arrow requires a positive training max_len")
        self.data = load_from_disk(datapath)
        if pretokenized_text_only:
            # External text Arrow may have no saved torch format. Select an
            # in-memory view without changing token values, masks, or row order.
            # Excluding messages also prevents HS requests from re-tokenizing
            # already encoded text through the Chat Completions API.
            self.data = self.data.with_format(
                "torch", columns=["input_ids", "loss_mask"], output_all_columns=False
            )
        self._select_split(split, split_ratio)

        self.transfer = transfer or FileTransfer(Path(datapath) / "hidden_states")
        self.vllm_endpoint = vllm_endpoint
        self.on_missing = on_missing
        self.on_generate = on_generate
        self.client: openai.OpenAI | None = None
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def _select_split(self, split, split_ratio):
        """Keep old signed-ratio and new named splits on the same boundary."""
        self.start_file_idx = 0
        if split is not None:
            if split not in ("train", "validation") or not 0.0 < split_ratio <= 1.0:
                raise ValueError(
                    "Named splits need train/validation and ratio in (0, 1)"
                )
            if split == "validation" and split_ratio == 1.0:
                raise ValueError("train_ratio=1.0 leaves no validation split")
            # Both views use this same positive ratio and integer boundary.
            # Reconstructing it as 1 + (ratio - 1) can round down by one row.
            split_idx = int(len(self.data) * split_ratio)
            if split_idx == 0 or (
                split == "validation" and split_idx == len(self.data)
            ):
                raise ValueError(f"{split} split is empty")
            if split == "train":
                self.data = self.data.select(range(split_idx))
            else:
                self.start_file_idx = split_idx
                self.data = self.data.select(range(split_idx, len(self.data)))
        elif split_ratio == 1.0:
            pass
        elif 1.0 > split_ratio > 0:
            self.start_file_idx = 0
            split_idx = int(len(self.data) * split_ratio)
            self.data = self.data.select(range(split_idx))
        elif -1.0 < split_ratio < 0:
            split_idx = int(len(self.data) * (1.0 + split_ratio))
            self.start_file_idx = split_idx
            self.data = self.data.select(range(split_idx, len(self.data)))
        else:
            raise ValueError("split_ratio must be in range (-1.0, 1.0] excluding 0.0.")

    def _map_to_file_idx(self, index: int):
        return index + self.start_file_idx

    def _setup_client(self):
        client = openai.OpenAI(
            base_url=self.vllm_endpoint, api_key="EMPTY", max_retries=0
        )
        list_models = client.models.list()
        model_id = list_models.data[0].id
        if self.model and self.model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({self.model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )
        self.model = model_id
        self.transfer.setup()
        self.client = client

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples."""
        return list(self.data.with_format(None)["seq_len"])

    def _get_dataset_item(self, index):
        item = self.data[index]
        if self.pretokenized_text_only:
            # DSpark consumes this prefix during collation anyway. Bound the HS
            # request too, without rewriting Arrow or changing sampler row IDs.
            item = {
                key: item[key][: self.max_len] for key in ("input_ids", "loss_mask")
            }
        return item

    def _align_text_hs(self, loaded_hs, input_ids, file_idx, *, allow_prefix):
        """Validate external text HS before caching or reusing a causal prefix."""
        tokens = loaded_hs["token_ids"]
        hidden = loaded_hs["hidden_states"]
        length = input_ids.shape[0]
        expected_ndim = 3  # [tokens, auxiliary slots + teacher, hidden width]
        minimum_slots = 2  # At least one auxiliary slot plus the teacher.
        if (
            tokens.ndim != 1
            or hidden.ndim != expected_ndim
            or hidden.shape[0] != tokens.shape[0]
            or hidden.shape[1] < minimum_slots
            or hidden.shape[2] < 1
            or tokens.shape[0] < length
            or (not allow_prefix and tokens.shape[0] != length)
            or not torch.equal(tokens[:length], input_ids)
        ):
            remedy = (
                "Use a separate HS directory when changing data or increasing "
                "the training length; existing caches are not overwritten."
                if allow_prefix
                else "Check the target HS response's token alignment and shape."
            )
            raise ValueError(
                f"External Arrow HS for row {file_idx} does not match the required "
                f"{length}-token training prefix (token IDs, length, or shape). "
                + remedy
            )
        # Views only: leave a longer cached file and its loaded tensors intact.
        prefix = {"token_ids": tokens[:length], "hidden_states": hidden[:length]}
        check_hidden_states(prefix, input_ids.tolist())
        return prefix

    def _generate_hidden_states_once(
        self,
        index: int,
        dataset_item: dict,
        client_item: ClientItem,
    ) -> dict[str, torch.Tensor]:
        handle: str | None = None
        try:
            if not self.client:
                self._setup_client()
            handle = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(handle)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            # Covers token/shape mismatches and non-finite values. The Mooncake
            # transfer performs manifest/checksum validation first.
            if self.pretokenized_text_only:
                loaded_hs = self._align_text_hs(
                    loaded_hs,
                    dataset_item["input_ids"],
                    self._map_to_file_idx(index),
                    allow_prefix=False,
                )
            else:
                loaded_hs = align_hidden_states(
                    loaded_hs,
                    dataset_item["input_ids"].tolist(),
                    allow_prefix="messages" in client_item,
                )

            file_idx = self._map_to_file_idx(index)
            if self.on_generate == "cache":
                self.transfer.cache(handle, file_idx)
            else:
                try:
                    self.transfer.delete(handle)
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Loaded a valid hidden-state sample but failed to delete "
                        "handle %s: %s",
                        handle,
                        cleanup_error,
                    )
            return loaded_hs
        except Exception:
            if handle is not None:
                try:
                    self.transfer.delete(handle)
                except Exception as cleanup_error:  # noqa: BLE001
                    logger.warning(
                        "Failed to clean generated hidden-state handle %s: %s",
                        handle,
                        cleanup_error,
                    )
            raise

    def _generate_recoverable_hs(self, index):
        dataset_item = self._get_dataset_item(index)
        client_item = build_client_item(dataset_item)
        # Token/shape errors on retokenized multimodal inputs remain fatal.
        if "messages" in client_item:
            return self._generate_hidden_states_once(index, dataset_item, client_item)
        return self.generation_recovery.run(
            lambda: self._generate_hidden_states_once(index, dataset_item, client_item),
            description=f"Hidden-state round trip failed for dataset index {index}",
        )

    def _maybe_generate_hs(self, index: int) -> dict[str, torch.Tensor] | None:
        if self.generation_recovery is not None and not self.pretokenized_text_only:
            return self._generate_recoverable_hs(index)
        if not self.client:
            self._setup_client()

        dataset_item = self._get_dataset_item(index)
        # Legacy HF templates kept the final assistant terminator intact.
        client_item = build_client_item(dataset_item, legacy_final_message=True)

        try:
            handle = generate_hidden_states(
                self.client,  # type:ignore[arg-type]
                self.model,  # type:ignore[arg-type]
                client_item,
                timeout=self.request_timeout,
                max_retries=self.max_retries,
            )

            loaded_hs = self.transfer.get_generated(handle)
            if loaded_hs is None:
                raise ValueError(f"Failed to load hidden states for handle {handle}")

            file_idx = self._map_to_file_idx(index)
            if self.pretokenized_text_only:
                loaded_hs = self._align_text_hs(
                    loaded_hs, dataset_item["input_ids"], file_idx, allow_prefix=False
                )
            else:
                loaded_hs = align_hidden_states(
                    loaded_hs,
                    dataset_item["input_ids"].tolist(),
                    allow_prefix="messages" in client_item,
                )

            match self.on_generate:
                case "cache":
                    self.transfer.cache(handle, file_idx)
                case "delete":
                    self.transfer.delete(handle)
        except Exception as e:
            if "messages" in client_item and isinstance(
                e, (ValueError, InvalidResponseError)
            ):
                # A retokenized MM response with the wrong prefix is not a
                # transient unavailable sample. Never silently train/skip it.
                raise
            if isinstance(e, ValueError) and (
                self.pretokenized_text_only or "NaN" in str(e)
            ):
                raise
            warnings.warn(
                f"Failed to load/cache hidden states for sample {index}: {e}",
                stacklevel=1,
            )
            return None

        return loaded_hs

    def _get_raw_data(self, index):
        file_idx = self._map_to_file_idx(index)
        loaded_hs = self.transfer.get_cached(file_idx)
        cached = loaded_hs is not None

        if loaded_hs is None:
            match self.on_missing:
                case "generate":
                    loaded_hs = self._maybe_generate_hs(index)
                case "skip":
                    return None
                case "warn":
                    warnings.warn(
                        f"Failed to load hidden states for sample {index}. Skipping...",
                        stacklevel=1,
                    )
                    return None
                case "raise":
                    raise RuntimeError(
                        f"Failed to load hidden states for sample {index}."
                    )

        if loaded_hs is None or isinstance(loaded_hs, SampleUnavailable):
            return loaded_hs

        dataset_item = self._get_dataset_item(index)
        if cached and self.pretokenized_text_only:
            loaded_hs = self._align_text_hs(
                loaded_hs, dataset_item["input_ids"], file_idx, allow_prefix=True
            )

        # loaded_hs structure: {
        #   "hidden_states": [seq_len, num_layers, hidden_size]
        #   "token_ids": [seq_len]
        # }

        try:
            loaded_hs = align_hidden_states(
                loaded_hs,
                dataset_item["input_ids"].tolist(),
                allow_prefix=_has_multimodal_content(dataset_item.get("messages", [])),
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Invalid hidden states for row {file_idx}: {exc}. "
                "Regenerate with data_generation_offline.py --validate-outputs "
                "to preserve invalid caches separately and retry them."
            ) from exc

        return {
            "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(
                1
            ),  # [seq_len, 3 * hidden_size]
            "input_ids": loaded_hs["token_ids"],  # [seq_len]
            "verifier_last_hidden_states": loaded_hs["hidden_states"][
                :, -1
            ],  # [seq_len, hidden_size]
            "loss_mask": dataset_item["loss_mask"],  # [seq_len]
        }


class SampleFileDataset(BaseDataset):
    def __init__(
        self,
        max_len: int,
        datapath: str | None = None,
        file_list: list[str] | None = None,
        transform: TransformTensors | None = None,
        hidden_states_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the SampleFileDataset.
        Args:
            max_len: The maximum length of the sequence.
            datapath: The path to the data directory. All `.pt` files in this directory
            or its subdirectories will be loaded and used as training data. MUTUALLY
            EXCLUSIVE with `file_list`.
            file_list: The list of explict file paths to load data from. These files
            must be in the format produced by the Speculators generation scripts.
            MUTUALLY EXCLUSIVE with `datapath`.
            transform: The transform to apply to the data.
            hidden_states_dtype: The dtype of the hidden states.
            standardize_fn: The function to standardize the data.

            Note: datapath or file_list must be provided, but not both.

        """

        if datapath is not None and file_list is not None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        if datapath is not None:
            file_list = list_files(datapath)

        if file_list is None:
            raise ValueError(
                "Either `datapath` or `file_list` must be provided, but "
                "not both. Use `datapath` to auto-discover files, or "
                "`file_list` to use a list of explicit file paths."
            )

        self.data: list[str] = file_list

        # Delay super init so that `_compute_approx_lengths` has required data
        super().__init__(max_len, transform, hidden_states_dtype)

    def __len__(self):
        return len(self.data)

    def _compute_approx_lengths(self) -> list[int]:
        """Get lengths of the dataset samples.

        First tries to load exact lengths from sample_lengths.json if available.
        Falls back to approximation based on file sizes.
        """
        # Look for the sample_lengths.json file
        sample_lengths_path = Path(self.data[0]).parent / "sample_lengths.json"
        if sample_lengths_path.exists():
            try:
                with sample_lengths_path.open() as f:
                    sample_lengths = json.load(f)
                # Extract file index from filename (e.g., data_42.pt -> 42)
                lengths = []
                for fname in self.data:
                    file_stem = Path(fname).stem
                    file_idx = file_stem.split("_")[-1]
                    lengths.append(sample_lengths[file_idx])
                return lengths
            except (KeyError, ValueError):
                pass

        # Fallback: approximate lengths from file sizes
        item_0 = self.__getitem__(0)
        if item_0 is None:
            raise ValueError(
                "Failed to load first element of datasets for length approximation"
            )
        lengths_0 = item_0["lengths"]
        # this is a single sample so there is only one length
        lengths_0 = lengths_0[0].item()
        size_0 = Path(self.data[0]).stat().st_size

        return [
            math.ceil(Path(fname).stat().st_size / size_0 * lengths_0)
            for fname in self.data
        ]

    def _get_raw_data(self, index):
        return standardize_data_v1(
            torch.load(
                self.data[index], mmap=True, weights_only=True, map_location="cpu"
            )
        )


class CollateFn:
    """Picklable collate function for use with ``multiprocessing_context='spawn'``."""

    def __init__(
        self,
        max_len: int,
        hidden_size: int,
        num_target_layers: int = 3,
        dtype: torch.dtype = torch.bfloat16,
        preprocess: Callable[[BatchType], BatchType] | None = None,
    ):
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_target_layers = num_target_layers
        self.dtype = dtype
        self.preprocess = preprocess

    def _clean_batch(
        self, batch: Sequence[BatchType | SampleUnavailable | None]
    ) -> tuple[list[BatchType], list[SampleUnavailable], int]:
        """Preprocess valid samples and collect unavailable and dropped samples."""
        preprocess = self.preprocess
        unavailable = []
        num_dropped = 0
        new_batch = []
        for item in batch:
            if item is None:
                num_dropped += 1
                continue
            if isinstance(item, SampleUnavailable):
                unavailable.append(item)
                num_dropped += 1
                continue

            new_batch.append(preprocess(item) if preprocess else item)

        return new_batch, unavailable, num_dropped

    def __call__(
        self, batch: Sequence[BatchType | SampleUnavailable | None]
    ) -> BatchType:
        max_len = self.max_len
        dtype = self.dtype

        batch, unavailable, num_dropped = self._clean_batch(batch)

        if not batch:
            # Create empty sample which then gets padded to full
            # batch size if no valid samples are found.
            # Match the configured `dtype` so the placeholder doesn't crash
            # downstream layers loaded at a different precision (e.g. bf16
            # weights vs fp32 default placeholders).
            empty = create_empty_sample(
                self.hidden_size, self.num_target_layers, dtype=dtype
            )
            if self.preprocess:
                empty = self.preprocess(empty)
            batch = [empty]
            locally_empty = True
        else:
            locally_empty = False

        collated_data: BatchType = {}
        for key in batch[0]:  # type: ignore[union-attr]
            if key == "lengths":
                collated_data[key] = torch.cat([b[key] for b in batch], dim=0)  # type: ignore[index]
                continue
            # one copy per sample: preallocated buffer, hidden states cast during write
            first = batch[0][key]  # type: ignore[index]
            buffer_dtype = dtype if "hidden_states" in key else first.dtype
            out = torch.zeros(
                (max_len, *first.shape[1:]), dtype=buffer_dtype, device=first.device
            )
            offset = 0
            for b in batch:
                tensor = b[key]  # type: ignore[index]
                num_rows = min(tensor.shape[0], max_len - offset)
                out[offset : offset + num_rows] = tensor[:num_rows]
                offset += num_rows
                if offset == max_len:
                    break
            collated_data[key] = out.unsqueeze(0)
            # shape: [1, max_len, ...]

        # Include lengths until while they fit in max_len
        # The last included length is (if necessary) truncated
        # Any additional lengths are discarded
        lengths = collated_data.pop("lengths")
        new_lengths = []
        cum_length = 0
        for length in lengths:
            if length + cum_length >= max_len:
                new_lengths.append(max_len - cum_length)
                break
            new_lengths.append(length)
            cum_length += length
        lengths = torch.tensor(new_lengths, dtype=torch.long)

        # Create document_ids: maps each position to its document index, -1 for padding
        document_ids = torch.repeat_interleave(
            torch.arange(lengths.shape[0], dtype=torch.long), lengths
        )
        document_ids = torch.cat(
            [
                document_ids,
                -1 * torch.ones(max_len - document_ids.shape[0], dtype=torch.long),
            ]
        ).unsqueeze(0)
        # shape: [1, max_len]
        collated_data["document_ids"] = document_ids

        collated_data["error_records"] = num_dropped
        metadata = RecoveryMetadata.from_unavailable(
            unavailable,
            locally_empty=locally_empty,
        )
        if metadata.failure_count or metadata.locally_empty:
            collated_data[RECOVERY_METADATA_KEY] = metadata

        return collated_data


def create_collate_fn(
    max_len: int,
    hidden_size: int,
    num_target_layers: int = 3,
    dtype: torch.dtype = torch.bfloat16,
    preprocess: Callable[[BatchType], BatchType] | None = None,
):
    """Compatibility factory for the upstream picklable collator."""
    return CollateFn(max_len, hidden_size, num_target_layers, dtype, preprocess)
