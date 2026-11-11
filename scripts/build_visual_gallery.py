"""Generate a visual comparison gallery: Real Faces vs Synthetic Faces vs Mined Hard Samples."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Automatic root resolution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.common.config import cfg_get, load_config, resolve
from src.common.logger import get_logger
from src.common.visualizer import plot_real_vs_synthetic_comparison

logger = get_logger(__name__)


def extract_paths_from_manifest(
    manifest_path: Path,
    max_count: int = 4,
    prioritize_modes: bool = False,
) -> List[str]:
    """Read existing image file paths from a JSONL manifest, optionally balancing manipulation modes."""
    if not manifest_path.is_file():
        return []

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                fp = item.get("file_path")
                if fp and Path(fp).exists():
                    items.append(item)
            except Exception:
                pass

    if not items:
        return []

    if prioritize_modes:
        # Group by mode and pick top-scoring or first from each mode
        by_mode: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            m = it.get("manipulation_type", "unknown")
            by_mode.setdefault(m, []).append(it)

        # Sort each mode group by hardness score if available
        for m_list in by_mode.values():
            m_list.sort(
                key=lambda x: (
                    (x.get("scores") or {}).get("real_manifold")
                    or x.get("baseline_score")
                    or 0.0
                ),
                reverse=True,
            )

        selected: List[str] = []
        modes = list(by_mode.keys())
        idx = 0
        while len(selected) < max_count and any(by_mode.values()):
            curr_mode = modes[idx % len(modes)]
            if by_mode[curr_mode]:
                selected.append(by_mode[curr_mode].pop(0)["file_path"])
            idx += 1
        return selected

    return [it["file_path"] for it in items[:max_count]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison gallery for Topic 11.")
    parser.add_argument("--config", type=str, default="configs/dataset_factory.yaml", help="Path to config YAML")
    parser.add_argument("--real-manifest", type=str, default=None)
    parser.add_argument("--synthetic-manifest", type=str, default=None)
    parser.add_argument("--hard-manifest", type=str, default=None)
    parser.add_argument("--output-image", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if Path(args.config).is_file():
        cfg = load_config(Path(args.config))

    real_manifest = Path(resolve(args.real_manifest, cfg, "paths.raw_manifest"))
    syn_manifest = Path(resolve(args.synthetic_manifest, cfg, "paths.curated_manifest"))
    hard_manifest = Path(resolve(args.hard_manifest, cfg, "paths.hard_manifest"))
    output_image = Path(resolve(args.output_image, cfg, "paths.gallery_image"))

    real_paths = extract_paths_from_manifest(real_manifest, args.num_samples)
    synthetic_paths = extract_paths_from_manifest(syn_manifest, args.num_samples, prioritize_modes=True)
    hard_paths = extract_paths_from_manifest(hard_manifest, args.num_samples, prioritize_modes=True)

    logger.info(
        "Extracted paths for gallery: %d real, %d synthetic, %d hard samples",
        len(real_paths),
        len(synthetic_paths),
        len(hard_paths),
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plot_real_vs_synthetic_comparison(
        real_paths=real_paths,
        synthetic_paths=synthetic_paths,
        hard_paths=hard_paths,
        output_path=output_image,
        num_samples=args.num_samples,
    )
    logger.info("Comparison gallery successfully saved to %s", output_image)


if __name__ == "__main__":
    main()
