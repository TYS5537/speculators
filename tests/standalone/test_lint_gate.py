"""Lint debt must not grow, disappear silently or hide tool failures."""

# ruff: noqa: PT009, PT027 -- This suite also runs with stdlib unittest.

import copy
import importlib.util
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "lint_gate", ROOT / "scripts/quality/check_lint.py"
)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class LintGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "sample.py"
        self.source.write_text(
            "class Model:\n    def forward(self):\n        pass\n", encoding="utf-8"
        )
        self.entry = {
            "path": "sample.py",
            "symbol": "Model.forward",
            "complexity": 12,
            "reason": "Split model orchestration after numerical regression coverage.",
        }
        self.baseline = {("sample.py", "Model.forward"): 12}

    def baseline_file(self, data=None):
        path = self.root / "baseline.json"
        if data is None:
            data = {"version": 1, "entries": [self.entry]}
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def finding(self, **overrides):
        result = {
            "filename": str(self.source),
            "location": {"row": 2, "column": 5},
            "code": "C901",
            "message": "`forward` is too complex (12 > 10)",
        }
        result.update(overrides)
        return result

    def test_baseline_loading_and_existing_debt(self):
        baseline = gate.load_baseline(self.baseline_file())
        self.assertEqual(baseline, self.baseline)
        self.assertEqual(gate.check_findings(self.root, baseline, [self.finding()]), [])
        self.assertEqual(gate.check_findings(self.root, {}, []), [])

    def test_line_shifts_do_not_change_identity(self):
        self.source.write_text(
            "# added comment\n\nclass Model:\n    def forward(self):\n        pass\n",
            encoding="utf-8",
        )
        findings = [self.finding(location={"row": 4, "column": 5})]
        self.assertEqual(gate.check_findings(self.root, self.baseline, findings), [])

    def test_classes_and_nested_async_functions_have_distinct_identities(self):
        source = (
            "def outer():\n"
            "    async def forward():\n        pass\n"
            "class Other:\n    def forward(self):\n        pass\n"
        )
        self.source.write_text(source, encoding="utf-8")
        baseline = {
            ("sample.py", "outer.forward"): 12,
            ("sample.py", "Other.forward"): 12,
        }
        findings = [
            self.finding(),
            self.finding(location={"row": 5, "column": 5}),
        ]
        self.assertEqual(gate.check_findings(self.root, baseline, findings), [])

    def test_new_noncomplexity_findings_always_fail(self):
        for code in ("F821", "S310", "I001", "invalid-syntax"):
            with self.subTest(code=code):
                errors = gate.check_findings(
                    self.root, {}, [self.finding(code=code, message="New lint finding")]
                )
                self.assertIn(code, errors[0])

    def test_new_or_increased_complexity_fails(self):
        errors = gate.check_findings(self.root, {}, [self.finding()])
        self.assertIn("new complexity debt", errors[0])
        errors = gate.check_findings(
            self.root,
            self.baseline,
            [self.finding(message="`forward` is too complex (13 > 10)")],
        )
        self.assertIn("reduce complexity", errors[0])

    def test_improvements_require_tightening_or_removing_the_baseline(self):
        findings = [self.finding(message="`forward` is too complex (11 > 10)")]
        errors = gate.check_findings(self.root, self.baseline, findings)
        self.assertIn("tighten baseline", errors[0])
        tighter = {("sample.py", "Model.forward"): 11}
        self.assertEqual(gate.check_findings(self.root, tighter, findings), [])
        errors = gate.check_findings(self.root, self.baseline, [])
        self.assertIn("remove stale", errors[0])

    def test_invalid_baseline_entries_fail_closed(self):
        invalid_fields = {
            "path": (
                "",
                "/outside.py",
                "../outside.py",
                "C:/x.py",
                "x\\y.py",
                "./x.py",
            ),
            "symbol": ("", "Model..forward", "not a name", None),
            "complexity": (True, 10, 12.5, "12"),
            "reason": ("", "  ", None),
        }
        for field, values in invalid_fields.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    entry = {**self.entry, field: value}
                    path = self.baseline_file({"version": 1, "entries": [entry]})
                    with self.assertRaises(ValueError):
                        gate.load_baseline(path)

    def test_unknown_schema_and_duplicates_fail_closed(self):
        for data in (
            [],
            {"version": 2, "entries": []},
            {"version": True, "entries": []},
            {"version": 1, "entries": {}},
            {"version": 1, "entries": [self.entry, self.entry]},
            {"version": 1, "entries": [{**self.entry, "code": "F821"}]},
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                gate.load_baseline(self.baseline_file(data))

    def test_malformed_diagnostics_fail_closed(self):
        invalid = (
            None,
            {},
            self.finding(location={"row": True, "column": 1}),
            self.finding(location={"row": 1, "column": 1}),
            self.finding(message="New complexity message format"),
            self.finding(message="`forward` is too complex (12 > 11)"),
            self.finding(message="`other` is too complex (12 > 10)"),
            self.finding(filename=str(self.root.parent / "outside.py")),
            self.finding(code=None),
        )
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                gate.check_findings(self.root, self.baseline, [item])
        with self.assertRaises(ValueError):
            gate.check_findings(self.root, self.baseline, {})
        with self.assertRaises(ValueError):
            gate.check_findings(self.root, self.baseline, [self.finding()] * 2)

    def test_runner_accepts_only_consistent_success_or_findings(self):
        for code, findings in ((0, []), (1, [self.finding()])):
            result = subprocess.CompletedProcess([], code, json.dumps(findings), "")
            with patch.object(gate.subprocess, "run", return_value=result) as run:
                self.assertEqual(gate.run_ruff(self.root), findings)
                self.assertEqual(run.call_args.kwargs["cwd"], self.root)
                self.assertEqual(
                    run.call_args.args[0][:3], [gate.sys.executable, "-m", "ruff"]
                )
                self.assertNotIn("--fix", run.call_args.args[0])

    def test_runner_rejects_tool_failure_or_unexpected_output(self):
        for code, stdout, stderr in (
            (2, "", "required-version mismatch"),
            (1, "[]", ""),
            (0, json.dumps([self.finding()]), ""),
            (0, "not json", ""),
            (0, "{}", ""),
        ):
            result = subprocess.CompletedProcess([], code, stdout, stderr)
            with (
                self.subTest(code=code, stdout=stdout),
                patch.object(gate.subprocess, "run", return_value=result),
                self.assertRaises(ValueError),
            ):
                gate.run_ruff(self.root)

    def test_cli_status_and_no_baseline_rewrite(self):
        path = self.baseline_file()
        before = path.read_bytes()
        findings = [self.finding()]
        for payload, status, message in (
            (findings, 0, "1 recorded C901 exceptions"),
            ([], 1, "remove stale"),
            ([self.finding(code="F821")], 1, "F821"),
        ):
            with (
                self.subTest(status=status, message=message),
                patch.object(gate, "BASELINE", path),
                patch.object(gate, "ROOT", self.root),
                patch.object(gate, "run_ruff", return_value=copy.deepcopy(payload)),
                patch.object(gate.sys, "stdout", new_callable=io.StringIO) as stdout,
                patch.object(gate.sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                self.assertEqual(gate.main(), status)
                self.assertIn(message, stdout.getvalue() + stderr.getvalue())
                self.assertEqual(path.read_bytes(), before)

    def test_cli_handles_missing_tools_and_timeouts(self):
        for error in (OSError("missing ruff"), subprocess.TimeoutExpired("ruff", 60)):
            with (
                self.subTest(error=error),
                patch.object(gate, "BASELINE", self.baseline_file()),
                patch.object(gate, "run_ruff", side_effect=error),
                patch.object(gate.sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                self.assertEqual(gate.main(), 2)
                self.assertIn("could not complete", stderr.getvalue())

    @unittest.skipUnless(importlib.util.find_spec("ruff"), "Install pinned Ruff first")
    def test_real_ruff_rejects_regressions_and_version_drift(self):
        config = self.root / "pyproject.toml"
        config.write_text(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        source = self.root / "scripts" / "sample.py"
        source.parent.mkdir()
        baseline = {("scripts/sample.py", "forward"): 12}
        for branches, extra, expected in (
            (11, "", None),
            (12, "", "reduce complexity"),
            (10, "", "tighten baseline"),
            (11, "\nmissing_symbol()\n", "F821"),
        ):
            with self.subTest(branches=branches, extra=extra):
                source.write_text(
                    "def forward(value):\n    result = 0\n"
                    + "    if value:\n        result += 1\n" * branches
                    + "    return result\n"
                    + extra,
                    encoding="utf-8",
                )
                # The initial setUp fixture is not part of this real Ruff scan.
                self.source.unlink(missing_ok=True)
                errors = gate.check_findings(
                    self.root, baseline, gate.run_ruff(self.root)
                )
                if expected is None:
                    self.assertEqual(errors, [])
                else:
                    self.assertTrue(any(expected in error for error in errors))
        config.write_text(
            '[tool.ruff]\nrequired-version = "==0.0.0"\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "Ruff failed"):
            gate.run_ruff(self.root)


if __name__ == "__main__":
    unittest.main()
