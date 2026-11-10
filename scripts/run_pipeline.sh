#!/usr/bin/env bash
# ==============================================================================
# Pipeline Execution Runner (Stable Diffusion + dinoViT Hard-Sample Mining Loop)
# Usage:
#   bash scripts/run_pipeline.sh [STAGE] [CONFIG_PATH]
# Allowed stages:
#   generate | curate | mine | train | eval | infer
# ==============================================================================

set -euo pipefail

STAGE="${1:-train}"
CONFIG="${2:-configs/config.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

export TMPDIR="${SCRIPT_DIR}/.cache/tmp"
mkdir -p "$TMPDIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"


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
    "clean-10k"|"production")
        echo "[Production] Launching Master Clean 10,000 Pipeline..."
        ${PYTHON} scripts/run_clean_10k_production_pipeline.py "${@:2}"
        ;;
    "block1"|"baseline")
        echo "[Stage 1] Running Block 1 Baseline Initialization..."
        ${PYTHON} -m src.pipelines.run_block1_baseline "${@:2}"
        ;;
    "active-learning"|"al")
        echo "[Stage 2-3] Running Closed-Loop Active Learning..."
        ${PYTHON} -m src.pipelines.run_closed_loop_active_learning "${@:2}"
        ;;
    "generate")
        echo "[Phase 1a] Generating synthetic DF images via Stable Diffusion..."
        ${PYTHON} -m src.pipelines.generate_deepfake_images "${@:2}"
        ;;
    "curate")
        echo "[Phase 1b] Filtering synthetic DF images via Curation Gate..."
        ${PYTHON} -m src.pipelines.curate_image_dataset "${@:2}"
        ;;
    "mine")
        echo "[Phase 2] Running baseline dinoViT to mine bypassed Hard Samples..."
        ${PYTHON} -m src.pipelines.mine_hard_samples "${@:2}"
        ;;
    "train")
        echo "[Phase 3a] Fine-tuning dinoViT on Baseline + Synthetic + Hard Samples..."
        ${PYTHON} -m src.pipelines.train --config "${CONFIG}" "${@:3}"
        ;;
    "eval"|"evaluate")
        echo "[Phase 3b] Evaluating dinoViT benchmark..."
        ${PYTHON} -m src.pipelines.evaluate --config "${CONFIG}" "${@:3}"
        ;;
    "infer"|"inference")
        echo "Launching dinoViT inference pipeline..."
        ${PYTHON} -m src.pipelines.infer --config "${CONFIG}"
        ;;
    *)
        echo "Error: Unknown stage '${STAGE}'. Allowed: clean-10k, block1, active-learning, generate, curate, mine, train, eval, infer."
        exit 1
        ;;
esac
