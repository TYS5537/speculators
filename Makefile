MDFILES := $(shell find . -name "*.md" -not -path "./.venv/*" -not -path "./build/*" -not -path "./.pytest_cache/*")
PYTHON ?= python
# Pick up new CPU Muse unit-test modules without maintaining a second file list.
MUSE_MODEL_TESTS := $(sort $(wildcard tests/unit/models/test_muse_*.py))
MUSE_TRAIN_TESTS := $(sort $(wildcard tests/unit/train/test_muse_*.py))
.DEFAULT_GOAL := quality

.PHONY: quality style test-fast test-muse

# No training stack or root pytest conftest; install standalone requirements first.
test-fast:
	$(PYTHON) -c "import datasets; import numpy; import pyarrow"
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests/standalone -v

# All Muse unit tests + shared checkpointing/training contracts; CPU dependencies.
test-muse:
	PYTHONPATH=src:hs_connectors/src $(PYTHON) -m pytest --noconftest -p no:cacheprovider -o addopts='' \
		$(MUSE_MODEL_TESTS) \
		tests/unit/models/test_activation_checkpointing.py \
		$(MUSE_TRAIN_TESTS) \
		tests/unit/train/test_cli_args.py \
		tests/unit/train/test_draft_config_init.py \
		tests/unit/train/test_rope_config.py \
		tests/unit/train/test_vocab_mapping_startup.py \
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
