"""Run pinned Ruff, allowing only the recorded, non-growing C901 debt.

This gate never edits code or refreshes its baseline. Fixes must tighten/remove
baseline entries; new lint findings and increased complexity are always errors.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "scripts/quality/lint_baseline.json"
_MAX_COMPLEXITY = 10
_COMPLEXITY_MESSAGE = re.compile(r"`([^`]+)` is too complex \((\d+) > (\d+)\)")
_DEFINITIONS = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _baseline_entry(entry: dict) -> tuple[tuple[str, str], int]:
    if not isinstance(entry, dict) or set(entry) != {
        "path",
        "symbol",
        "complexity",
        "reason",
    }:
        raise ValueError("Baseline entries need path, symbol, complexity and reason")
    path, symbol = entry["path"], entry["symbol"]
    if not isinstance(path, str) or not path:
        raise ValueError("Baseline paths must be nonempty repository-relative strings")
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in path
        or ":" in path
        or relative.as_posix() != path
        or relative.suffix != ".py"
    ):
        raise ValueError(f"Invalid baseline path: {path!r}")
    if not isinstance(symbol, str) or not all(
        part.isidentifier() for part in symbol.split(".")
    ):
        raise ValueError(f"Invalid qualified function name: {symbol!r}")
    complexity = entry["complexity"]
    if type(complexity) is not int or complexity <= _MAX_COMPLEXITY:
        raise ValueError("Baseline complexity must be an integer above 10")
    if not isinstance(entry["reason"], str) or not entry["reason"].strip():
        raise ValueError("Every complexity exception needs a reason")
    return (path, symbol), complexity


def load_baseline(path: Path) -> dict[tuple[str, str], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or set(data) != {"version", "entries"}
        or type(data["version"]) is not int
        or data["version"] != 1
        or not isinstance(data["entries"], list)
    ):
        raise ValueError("Expected baseline version 1 with an entries list")
    baseline = {}
    for entry in data["entries"]:
        key, complexity = _baseline_entry(entry)
        if key in baseline:
            raise ValueError(f"Duplicate baseline entry: {key}")
        baseline[key] = complexity
    return baseline


def _function_symbols(node: ast.AST, scope: tuple[str, ...] = ()):
    """Include class/nested-function scopes, independently of source line shifts."""
    for child in ast.iter_child_nodes(node):
        child_scope = (*scope, child.name) if isinstance(child, _DEFINITIONS) else scope
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child.lineno, ".".join(child_scope)
        yield from _function_symbols(child, child_scope)


def _diagnostic(root: Path, item: dict) -> tuple[str, int, int, str, str]:
    if not isinstance(item, dict):
        raise ValueError("Expected a Ruff diagnostic object")
    location = item.get("location")
    if not isinstance(location, dict):
        raise ValueError("Ruff diagnostic is missing its location")
    row, column = location.get("row"), location.get("column")
    if any(type(value) is not int or value < 1 for value in (row, column)):
        raise ValueError("Invalid Ruff source location")
    filename, code, message = (item.get(key) for key in ("filename", "code", "message"))
    if not all(isinstance(value, str) and value for value in (filename, code, message)):
        raise ValueError("Ruff diagnostic needs filename, code and message")
    path = Path(filename).resolve().relative_to(root.resolve()).as_posix()
    return path, row, column, code, message


def _complexity_key(root: Path, path: str, row: int, message: str, symbols: dict):
    match = _COMPLEXITY_MESSAGE.fullmatch(message)
    if match is None or int(match[3]) != _MAX_COMPLEXITY:
        raise ValueError(f"Unrecognized complexity diagnostic: {message}")
    if path not in symbols:
        tree = ast.parse((root / path).read_text(encoding="utf-8-sig"))
        symbols[path] = dict(_function_symbols(tree))
    symbol = symbols[path].get(row)
    if symbol is None or symbol.rsplit(".", 1)[-1] != match[1]:
        raise ValueError(f"Cannot locate the diagnosed function: {path}:{row}")
    return (path, symbol), int(match[2])


def check_findings(root: Path, baseline: dict, findings: list) -> list[str]:
    """Return all regressions and stale exceptions; malformed input fails closed."""
    if not isinstance(findings, list):
        raise ValueError("Expected a list of Ruff diagnostics")
    errors, seen, symbols = [], set(), {}
    for item in findings:
        path, row, column, code, message = _diagnostic(root, item)
        label = f"{path}:{row}:{column}: {code} {message}"
        if code != "C901":
            errors.append(label)
            continue
        key, complexity = _complexity_key(root, path, row, message, symbols)
        if key in seen:
            raise ValueError(f"Ambiguous or duplicate complexity diagnostic: {key}")
        seen.add(key)
        expected = baseline.get(key)
        if expected is None:
            errors.append(f"{label} (new complexity debt)")
        elif complexity != expected:
            action = (
                "reduce complexity" if complexity > expected else "tighten baseline"
            )
            errors.append(f"{label} (baseline {expected}; {action})")
    for path, symbol in sorted(baseline.keys() - seen):
        errors.append(f"{path}: {symbol}: remove stale complexity baseline entry")
    return errors


def run_ruff(root: Path) -> list:
    # required-version in pyproject.toml rejects tool-version drift before scanning.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--no-cache",
            "--output-format",
            "json",
            ".",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise ValueError(f"Ruff failed ({result.returncode}): {result.stderr.strip()}")
    findings = json.loads(result.stdout)
    if not isinstance(findings, list) or result.returncode != int(bool(findings)):
        raise ValueError("Ruff exit status and diagnostic list disagree")
    return findings


def main() -> int:
    try:
        baseline = load_baseline(BASELINE)
        errors = check_findings(ROOT, baseline, run_ruff(ROOT))
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as exc:
        print(f"Lint gate could not complete: {exc}", file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Lint gate passed; {len(baseline)} recorded C901 exceptions unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
