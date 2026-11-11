# ==============================================================================
# Makefile for UTH Computer Vision Coursework
# ==============================================================================

.PHONY: help install install-dev install-all setup-hooks format lint type-check test test-smoke clean run

SHELL := /bin/bash

# Auto-detect virtual environment
ifneq ($(VIRTUAL_ENV),)
    PYTHON := $(VIRTUAL_ENV)/bin/python
    PYTEST := $(VIRTUAL_ENV)/bin/pytest
    RUFF := $(VIRTUAL_ENV)/bin/ruff
    MYPY := $(VIRTUAL_ENV)/bin/mypy
else ifneq ($(wildcard .venv/bin/python),)
    PYTHON := .venv/bin/python
    PYTEST := .venv/bin/pytest
    RUFF := .venv/bin/ruff
    MYPY := .venv/bin/mypy
else
    PYTHON := python3
    PYTEST := pytest
    RUFF := ruff
    MYPY := mypy
endif

help:
	@echo "UTH Computer Vision Coursework Automation Commands:"
	@echo "  make install        Install base requirements"
	@echo "  make install-dev    Install dev, test, and tracking dependencies"
	@echo "  make install-all    Install editable package with all extras"
	@echo "  make setup-hooks    Install pre-commit hooks into git repository"
	@echo "  make format         Auto-format code using Ruff and Black"
	@echo "  make lint           Check code quality and style using Ruff"
	@echo "  make type-check     Run static type checker with Mypy"
	@echo "  make test           Run all unit tests with pytest"
	@echo "  make test-smoke     Run quick sanity smoke tests (< 5s)"
	@echo "  make clean          Clean temporary files, caches, and build artifacts"
	@echo "  make run            Run default pipeline training"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/base.txt -r requirements/deeplearning.txt

install-dev:
	$(PYTHON) -m pip install -r requirements/base.txt -r requirements/deeplearning.txt -r requirements/dev.txt -r requirements/tracking.txt

install-all:
	$(PYTHON) -m pip install -e .

setup-hooks:
	pre-commit install

format:
	@command -v $(RUFF) > /dev/null 2>&1 && $(RUFF) format src tests scripts && $(RUFF) check --fix src tests scripts || echo "Ruff not found, skipping auto-format"

lint:
	@if command -v $(RUFF) > /dev/null 2>&1; then \
		$(RUFF) check src tests scripts && $(RUFF) format --check src tests scripts; \
	elif command -v ruff > /dev/null 2>&1; then \
		ruff check src tests scripts && ruff format --check src tests scripts; \
	else \
		echo "Ruff not installed in environment. Running AST compilation sanity check..."; \
		$(PYTHON) -m compileall -q src tests scripts || exit 1; \
		echo "✅ All source files successfully verified for syntax and compilation."; \
	fi

type-check:
	@if command -v $(MYPY) > /dev/null 2>&1; then \
		$(MYPY) src tests; \
	elif command -v mypy > /dev/null 2>&1; then \
		mypy src tests; \
	else \
		echo "Mypy not installed in environment. Running module import validation..."; \
		$(PYTHON) -c "import src.common.metrics; import src.common.visualizer; import src.data.splits; import src.models.detector.image_detector; print('✅ Core modules type-load verified.')" || exit 1; \
	fi

test:
	$(PYTEST) tests/ -v

test-smoke:
	$(PYTEST) tests/ -m smoke -v

run:
	bash scripts/run_pipeline.sh train configs/config.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	rm -rf .cache/tmp/* build/ dist/ *.egg-info
	find . -type f -name "*.so" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Workspace cleaned successfully!"
