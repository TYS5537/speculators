MDFILES := $(shell find . -name "*.md" -not -path "./.venv/*" -not -path "./build/*" -not -path "./.pytest_cache/*")
PYTHON ?= python
.DEFAULT_GOAL := quality

.PHONY: quality style test-fast test-muse

# No training stack or root pytest conftest; install standalone requirements first.
test-fast:
	$(PYTHON) -c "import datasets; import numpy; import pyarrow"
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/standalone -v

# Real CPU model + training lifecycle; requires project dependencies and pytest.
test-muse:
	PYTHONPATH=src:hs_connectors/src $(PYTHON) -m pytest --noconftest -p no:cacheprovider -o addopts='' \
		tests/unit/models/test_muse_architecture.py \
		tests/unit/models/test_muse_anchor_correction.py \
		tests/unit/models/test_muse_core.py \
		tests/unit/models/test_muse_rollout_inputs.py \
		tests/unit/train/test_muse_config_contract.py \
		tests/unit/train/test_cli_args.py \
		tests/unit/train/test_draft_config_init.py \
		tests/integration/train/test_muse_training_resume.py

# run checks on all files for the repo
quality:
	@echo "Running quality checks"
	ruff check
	ruff format --check
	python -m mdformat --check $(MDFILES)
	mypy --check-untyped-defs

# style the code according to accepted standards for the repo
# Note: We run `ruff format` twice. Once to fix long lines before lint check
# and again to fix any formatting issues introduced by ruff check --fix
style:
	@echo "Running style fixes"
	ruff format
	ruff check --fix
	ruff format --silent
	python -m mdformat $(MDFILES)
