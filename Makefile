MDFILES := $(shell find . -name "*.md" -not -path "./.venv/*" -not -path "./build/*" -not -path "./.pytest_cache/*")
PYTHON ?= python
# Pick up new CPU MMuse unit-test modules without maintaining a second file list.
MMUSE_MODEL_TESTS := $(sort $(wildcard tests/unit/models/test_mmuse_*.py))
MMUSE_TRAIN_TESTS := $(sort $(wildcard tests/unit/train/test_mmuse_*.py))
.DEFAULT_GOAL := quality

.PHONY: lint lint-strict quality style test-fast test-mmuse test-muse

# Compatibility alias for existing local test commands.
test-muse: test-mmuse

# No training stack or root pytest conftest; install standalone requirements first.
test-fast:
	$(PYTHON) -c "import datasets; import numpy; import pyarrow"
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/standalone -v

# All MMuse unit tests + shared checkpointing/training contracts; CPU dependencies.
test-mmuse:
	PYTHONPATH=src:hs_connectors/src $(PYTHON) -m pytest --noconftest -p no:cacheprovider -o addopts='' \
		$(MMUSE_MODEL_TESTS) \
		tests/unit/models/test_activation_checkpointing.py \
		$(MMUSE_TRAIN_TESTS) \
		tests/unit/train/test_cli_args.py \
		tests/unit/train/test_draft_config_init.py \
		tests/unit/train/test_rope_config.py \
		tests/unit/train/test_vocab_mapping_startup.py \
		tests/integration/train/test_mmuse_training_resume.py

# Lightweight gate: no training dependencies, no new or growing lint debt.
lint:
	$(PYTHON) scripts/quality/check_lint.py
	$(PYTHON) -m ruff format --check

# Show every finding, including the documented C901 refactoring backlog.
lint-strict:
	$(PYTHON) -m ruff check
	$(PYTHON) -m ruff format --check

# run checks on all files for the repo
quality: lint
	@echo "Running quality checks"
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
