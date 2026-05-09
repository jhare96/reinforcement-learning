# Convenience targets that mirror the GitHub Actions CI workflow
# (.github/workflows/ci.yml + docs.yml). Run `make ci` locally before
# pushing to catch the same failures the pipeline does.

PYTHON ?= python
# Modules mypy is expected to keep strictly typed (mirrors
# TYPECHECK_PATHS in .github/workflows/ci.yml).
TYPECHECK_PATHS ?= rlib/envs rlib/utils/schedulers.py

.PHONY: help install lint format format-check typecheck test build docs ci clean

help:
	@echo "Targets:"
	@echo "  install       Editable install with classic + dev extras"
	@echo "  lint          ruff check ."
	@echo "  format        ruff format . (writes changes)"
	@echo "  format-check  ruff format --check ."
	@echo "  typecheck     mypy \$$(TYPECHECK_PATHS)"
	@echo "  test          pytest -ra --color=yes"
	@echo "  build         python -m build + twine check"
	@echo "  docs          mkdocs build --strict"
	@echo "  ci            lint + format-check + typecheck + test + build (matches ci.yml)"
	@echo "  clean         remove build artefacts and caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[classic,dev,docs]"
	$(PYTHON) -m pip install "mypy>=1.8" build twine

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy $(TYPECHECK_PATHS)

test:
	$(PYTHON) -m pytest -ra --color=yes

build:
	rm -rf dist
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

docs:
	$(PYTHON) -m mkdocs build --strict --site-dir site

# Mirrors ci.yml job order: lint -> typecheck -> test -> build.
ci: lint format-check typecheck test build

clean:
	rm -rf dist build site .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
