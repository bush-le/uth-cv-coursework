#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Script for UTH Computer Vision Coursework
# Usage:
#   bash scripts/setup_env.sh
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== Initializing Python Virtual Environment (.venv) ==="
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo "Virtual environment created at .venv"
else
    echo "Virtual environment already exists."
fi

# Source virtualenv
source .venv/bin/activate

echo "=== Upgrading pip, setuptools, wheel ==="
pip install --upgrade pip setuptools wheel

echo "=== Installing project dependencies ==="
pip install -r requirements.txt
pip install -r requirements/deeplearning.txt
pip install -r requirements/tracking.txt

echo "=== Installing pre-commit hooks ==="
if command -v pre-commit &>/dev/null; then
    pre-commit install || true
    echo "Pre-commit hooks successfully installed."
fi

echo "=== Environment setup complete! ==="
