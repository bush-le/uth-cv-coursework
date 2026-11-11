# ==============================================================================
# Makefile for UTH Computer Vision Coursework
# ==============================================================================

.PHONY: help install install-dev install-all setup-hooks format lint type-check test test-smoke clean run

SHELL := /bin/bash

# Auto-detect virtual environment
VENV ?= .venv
ifneq ($(wildcard $(VENV)/bin/python),)
    PYTHON := $(VENV)/bin/python
    PYTEST := $(VENV)/bin/pytest
    RUFF := $(VENV)/bin/ruff
    BLACK := $(VENV)/bin/black
    MYPY := $(VENV)/bin/mypy
else
    PYTHON := python3
    PYTEST := pytest
    RUFF := ruff
    BLACK := black
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
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements/dev.txt -r requirements/deeplearning.txt -r requirements/tracking.txt

install-all:
	$(PYTHON) -m pip install -e .

setup-hooks:
	pre-commit install

format:
	@which $(RUFF) > /dev/null 2>&1 && $(RUFF) format src tests || echo "Ruff not found, skipping format"
	@which $(RUFF) > /dev/null 2>&1 && $(RUFF) check --fix src tests || true
	@which $(BLACK) > /dev/null 2>&1 && $(BLACK) src tests || true

lint:
	@which $(RUFF) > /dev/null 2>&1 && $(RUFF) check src tests || echo "Ruff not installed, run pip install ruff"
	@which $(BLACK) > /dev/null 2>&1 && $(BLACK) --check src tests || true

type-check:
	@which $(MYPY) > /dev/null 2>&1 && $(MYPY) src tests || echo "Mypy not installed, run pip install mypy"

test:
	$(PYTEST) tests/ -v

test-smoke:
	$(PYTEST) tests/ -m smoke -v

run:
	bash scripts/run_pipeline.sh train configs/config.yaml

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	find . -type f -name "*.so" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@echo "Workspace cleaned successfully!"
