"""Read-only structural checks for explicitly trusted, externally encoded Arrow.

Valid token IDs cannot prove tokenizer/template provenance. The caller must opt
in and attest to that provenance; this never manufactures a native data manifest.
"""

from pathlib import Path

_COLUMNS = ("input_ids", "loss_mask", "seq_len")


def _validate_schema(schema, pa):
    missing = set(_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"External DSV4 Arrow is missing columns: {sorted(missing)}")
    for name in ("input_ids", "loss_mask"):
        kind = schema.field(name).type
        if not (
            pa.types.is_list(kind)
            or pa.types.is_large_list(kind)
            or pa.types.is_fixed_size_list(kind)
        ):
            raise ValueError(f"External DSV4 {name} must be a one-dimensional list")
        scalar = kind.value_type
        valid = pa.types.is_integer(scalar)
        if name == "loss_mask":
            valid = valid or pa.types.is_boolean(scalar) or pa.types.is_floating(scalar)
        if not valid:
            raise ValueError(
                f"External DSV4 {name} has unsupported value type {scalar}"
            )
    if not pa.types.is_integer(schema.field("seq_len").type):
        raise ValueError("External DSV4 seq_len must be an integer")


def _validate_batch(batch, vocab_size, start, pc):
    location = f"External DSV4 Arrow rows [{start}, {start + len(batch)})"
    ids = pc.list_flatten(batch["input_ids"])
    mask = pc.list_flatten(batch["loss_mask"])
    for name, values in (
        *((name, batch[name]) for name in _COLUMNS),
        ("input_ids", ids),
        ("loss_mask", mask),
    ):
        if values.null_count:
            raise ValueError(f"{location}: {name} contains null values")
    lengths = pc.list_value_length(batch["input_ids"])
    if pc.any(pc.less_equal(lengths, 0)).as_py():
        raise ValueError(f"{location}: input_ids must not be empty")
    if pc.any(pc.not_equal(lengths, pc.list_value_length(batch["loss_mask"]))).as_py():
        raise ValueError(f"{location}: input_ids and loss_mask lengths differ")
    try:
        seq_len = pc.cast(batch["seq_len"], "int64", safe=True)
    except ValueError as exc:
        raise ValueError(
            f"{location}: seq_len outside supported integer range"
        ) from exc
    if pc.any(pc.not_equal(lengths, seq_len)).as_py():
        raise ValueError(f"{location}: seq_len differs from input_ids length")
    if pc.min(ids).as_py() < 0 or pc.max(ids).as_py() >= vocab_size:
        raise ValueError(f"{location}: input_ids outside target vocabulary")
    # A temporary float64 view supports bool/integer/half masks uniformly. Exact
    # 0/1 survive casting; even rounded large uint64 values remain invalid. This
    # does not rewrite dataset values/dtypes. NaN/Inf fail these comparisons too.
    mask = pc.cast(mask, "float64", safe=False)
    if pc.any(pc.and_(pc.not_equal(mask, 0), pc.not_equal(mask, 1))).as_py():
        raise ValueError(f"{location}: loss_mask must contain only finite 0/1 values")


def validate_external_arrow(path, vocab_size, *, rank=0, world_size=1, batch_size=256):
    """Scan a rank's contiguous slice, with bounded memory and no dataset writes.

    The caller must synchronize failures from every rank before building the
    model. Together those slices cover every row exactly once, including when
    there are fewer rows than ranks. No sampling, filtering or retokenizing.
    """
    if (
        any(
            type(value) is not int
            for value in (rank, world_size, batch_size, vocab_size)
        )
        or world_size < 1
        or not 0 <= rank < world_size
        or batch_size < 1
        or vocab_size < 1
    ):
        raise ValueError("Invalid external Arrow rank/world_size/batch_size/vocab_size")

    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.compute as pc  # noqa: PLC0415
    from datasets import Dataset, load_from_disk  # noqa: PLC0415

    source = Path(path).resolve()
    if not source.is_dir():
        raise ValueError("External DSV4 Arrow needs a Dataset.save_to_disk directory")
    try:
        dataset = load_from_disk(str(source))
    except (OSError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid or empty external DSV4 Dataset.save_to_disk directory: {source}"
        ) from exc
    if not isinstance(dataset, Dataset):
        raise ValueError("External DSV4 Arrow must be one Dataset, not a DatasetDict")
    if not len(dataset):
        raise ValueError("External DSV4 Arrow Dataset must not be empty")
    _validate_schema(dataset.features.arrow_schema, pa)
    start, stop = (
        len(dataset) * rank // world_size,
        len(dataset) * (rank + 1) // world_size,
    )
    # with_format/select only create views. Respect logical row order, including
    # saved indices, and ignore any saved numpy/torch formatting during checking.
    view = dataset.with_format("arrow", columns=list(_COLUMNS))
    checked = 0
    if start < stop:
        for batch in view.select(range(start, stop)).iter(batch_size=batch_size):
            _validate_batch(batch, vocab_size, start + checked, pc)
            checked += len(batch)
    return {
        "data_path": str(source),
        "row_count": len(dataset),
        "row_start": start,
        "row_stop": stop,
        "checked_rows": checked,
        "provenance": "user-asserted",
        "validation": "structure-only",
    }
