#!/usr/bin/env bash
# ==============================================================================
# Script to Download Pretrained Model Weights & Backbones
# Usage:
#   bash scripts/download_weights.sh [MODEL_NAME]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${SCRIPT_DIR}/experiments/checkpoints/pretrained"
mkdir -p "${TARGET_DIR}"

MODEL="${1:-default}"

echo "=== Downloading weights for: ${MODEL} ==="
echo "Target directory: ${TARGET_DIR}"

case "${MODEL}" in
    "default")
        echo "Placeholder for default backbone weights."
        ;;
    *)
        echo "Specified model: ${MODEL}"
        ;;
esac

echo "Download completed."
