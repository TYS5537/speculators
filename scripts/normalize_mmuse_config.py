"""Normalize MMuse checkpoint config metadata without reading model weights.

Print normalized JSON by default. Use --output for a separate config file, or
--in-place to replace config.json after saving its original bytes in a backup.
"""

import argparse
import json
from pathlib import Path

from speculators.models.mmuse.compat import normalize_mmuse_checkpoint_config
from speculators.provenance import atomic_write


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, help="Checkpoint directory or JSON file"
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output", type=Path, help="Write a new JSON file")
    destination.add_argument(
        "--in-place", action="store_true", help="Back up and replace the input config"
    )
    args = parser.parse_args(argv)
    source = args.checkpoint
    if source.is_dir():
        source = source / "config.json"
    try:
        original = source.read_bytes()
        normalized = normalize_mmuse_checkpoint_config(json.loads(original))
        content = (
            json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if args.in_place:
            if original.decode("utf-8-sig").replace("\r\n", "\n") == content:
                print(f"Already normalized: {source}")
                return
            backup = source.with_name(source.name + ".before-normalize.bak")
            with backup.open("xb") as handle:
                handle.write(original)
            atomic_write(source, content)
            print(f"Updated: {source}\nOriginal config: {backup}")
        elif args.output:
            if args.output.resolve() == source.resolve():
                raise ValueError("Use --in-place to replace the source with a backup.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
            print(f"Wrote: {args.output}")
        else:
            print(content, end="")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
