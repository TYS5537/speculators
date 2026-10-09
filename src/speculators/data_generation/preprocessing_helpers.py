"""Formatting helpers shared by render-based and offline preprocessing."""

import json
from pathlib import Path


def _adapt_part_for_vllm(part: str | dict):
    if isinstance(part, str):
        return {"type": "text", "text": part}

    part_type = part["type"]

    if part_type == "text":
        return {"type": "text", "text": part["text"]}

    for modality in ("image", "video", "audio"):
        if part_type == modality:
            if local_path := part.get("path"):
                file_url = f"file://{Path(local_path).absolute()}"
                return {"type": f"{modality}_url", f"{modality}_url": {"url": file_url}}
            if url := part.get("url"):
                return {"type": f"{modality}_url", f"{modality}_url": {"url": url}}

            if part.get("base64"):
                expr = {"type": modality, "base64": "..."}
                raise ValueError(
                    f"Content part {expr} is not supported. To avoid copying "
                    f"the {modality} when saving the preprocessed dataset, "
                    f"please express {modality} inputs using file paths or URLs."
                )
            if part.get(modality):
                expr = {"type": modality, modality: "..."}
                raise ValueError(
                    f"Content part {expr} is not supported. To avoid copying "
                    f"the {modality} when saving the preprocessed dataset, "
                    f"please express {modality} inputs using file paths or URLs."
                )

            expr = {"type": modality} | {k: "..." for k in part if k != "type"}
            raise NotImplementedError(f"Unknown content part: {expr}")

    expr = dict.fromkeys(part.keys(), "...")
    raise NotImplementedError(f"Unknown content part: {expr}")


def _adapt_turn_for_vllm(turn: dict):
    if isinstance(turn["content"], str):
        return turn

    return turn | {"content": [_adapt_part_for_vllm(part) for part in turn["content"]]}


def _adapt_conv_for_vllm(normalized_conv: list[dict]):
    return [_adapt_turn_for_vllm(turn) for turn in normalized_conv]


def visualize_sample(preprocessed, processor, log, idx: int = 0):
    """Visualize a single sample with color-coded trainable regions."""
    # Get preprocessed sample
    prep_sample = preprocessed[idx]
    input_ids = prep_sample["input_ids"].tolist()
    loss_mask = prep_sample["loss_mask"].tolist()

    log.info(f"SAMPLE #{idx}")
    log.info("HIGHLIGHTED TEXT (BLUE = trainable, GREY = masked)")

    # Create color-highlighted text
    blue = "\033[38;5;153m"  # Very light blue text for trainable tokens
    grey = "\033[90m"  # Grey text for masked tokens
    reset = "\033[0m"  # Reset color

    output = []
    prev_state = None

    for i in range(len(input_ids)):
        is_train = loss_mask[i] == 1
        token = processor.decode([input_ids[i]])
        assert isinstance(token, str)

        # Switch colors when state changes
        if is_train != prev_state:
            output.append(blue if is_train else grey)
            prev_state = is_train

        output.append(token)

    output.append(reset)
    highlighted = "".join(output)

    log.info(highlighted)


def parse_conv_tools(conv_tools: object, idx: int, log) -> list | None:
    """Parse the tools JSON string for one conversation; warn and return None
    on invalid JSON or unexpected types."""
    if not conv_tools:
        return None
    if isinstance(conv_tools, list):
        return conv_tools
    if not isinstance(conv_tools, str):
        log.warning(
            f"Non-string value in tools column for conversation {idx}: "
            f"{type(conv_tools).__name__}, proceeding without tools"
        )
        return None
    try:
        return json.loads(conv_tools)
    except json.JSONDecodeError as e:
        log.warning(
            f"Invalid JSON in tools column for conversation {idx}: {e}, "
            "proceeding without tools"
        )
        return None
