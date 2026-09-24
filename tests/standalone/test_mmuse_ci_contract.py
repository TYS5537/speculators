"""The CPU test recipe must remain installable after upstream dependency changes."""

# ruff: noqa: PT009 -- This suite also runs with stdlib unittest.

import sys
import unittest
from pathlib import Path

from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = "tests/mmuse_cpu_requirements.txt"


class MMuseCIContractTests(unittest.TestCase):
    def test_numerical_pins_satisfy_upstream_dependency_ranges(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        upstream = {
            req.name: req
            for raw in metadata["tool"]["speculators"]["dependencies"]["base"]
            for req in [Requirement(raw)]
        }
        baseline = {
            req.name: req
            for raw in (ROOT / REQUIREMENTS).read_text(encoding="utf-8").splitlines()
            if raw.strip() and not raw.startswith("#")
            for req in [Requirement(raw)]
        }
        for name in ("torch", "transformers"):
            with self.subTest(package=name):
                pins = list(baseline[name].specifier)
                self.assertEqual(len(pins), 1)
                self.assertEqual(pins[0].operator, "==")
                self.assertIn(pins[0].version, upstream[name].specifier)

    def test_ci_and_documented_install_share_the_same_requirements(self):
        command = f"uv pip install ./hs_connectors . -r {REQUIREMENTS}"
        for path in (".github/workflows/mmuse-tests.yml", "CONTRIBUTING.md"):
            with self.subTest(path=path):
                text = (ROOT / path).read_text(encoding="utf-8")
                self.assertIn(command, text)
                self.assertNotIn("transformers==", text)

    def test_cpu_ci_runs_tensorboard_logging_regressions(self):
        requirements = (ROOT / REQUIREMENTS).read_text(encoding="utf-8")
        self.assertIn("tensorboard==2.21.0", requirements.splitlines())
        self.assertIn(
            "tests/unit/train/test_logger.py",
            (ROOT / "Makefile").read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
