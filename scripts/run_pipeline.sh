#!/usr/bin/env bash
# ==============================================================================
# Pipeline Execution Runner
# Usage:
#   bash scripts/run_pipeline.sh [STAGE] [CONFIG_PATH]
# Examples:
#   bash scripts/run_pipeline.sh train configs/config.yaml
#   bash scripts/run_pipeline.sh eval configs/config.yaml
# ==============================================================================

set -euo pipefail

STAGE="${1:-train}"
CONFIG="${2:-configs/config.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [[ -f ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "=================================================="
echo "UTH Computer Vision Coursework - Pipeline Runner"
echo "Stage: ${STAGE}"
echo "Config: ${CONFIG}"
echo "Python: $(${PYTHON} --version)"
echo "=================================================="

case "${STAGE}" in
    "train")
        echo "Launching training pipeline..."
        ${PYTHON} -m src.pipelines.train --config "${CONFIG}"
        ;;
    "eval"|"evaluate")
        echo "Launching evaluation pipeline..."
        ${PYTHON} -m src.pipelines.evaluate --config "${CONFIG}"
        ;;
    "infer"|"inference")
        echo "Launching inference pipeline..."
        ${PYTHON} -m src.pipelines.infer --config "${CONFIG}"
        ;;
    *)
        echo "Error: Unknown stage '${STAGE}'. Allowed: train, eval, infer."
        exit 1
        ;;
esac
