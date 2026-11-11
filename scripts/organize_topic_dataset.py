"""Organize and standardize dataset structure strictly adhering to Topic 11 lifecycle.

Packages the generated, curated, and mined faces into standardized directory hierarchies,
assigns identity-disjoint splits, builds comparative benchmark stages (D0, D1, D2),
and auto-generates dataset analytics.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional

# Automatic root resolution
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.common.config import cfg_get, load_config, resolve
from src.common.logger import get_logger
from src.common.seed import seed_everything
from src.data.image_schema import ImageMetadata
from src.data.splits import (
    assign_identity_splits,
    build_stage,
    compute_dataset_stats,
)

logger = get_logger(__name__)


def organize_synthetic_faces(
    source_manifest: Path,
    output_base: Path,
) -> Dict[str, int]:
    """Organize synthetic faces into subfolders by generation method."""
    counts = {"text2image": 0, "inpainting": 0, "img2img": 0, "total": 0}
    if not source_manifest.is_file():
        return counts

    t2i_dir = output_base / "text2image"
    inpaint_dir = output_base / "inpainting"
    img2img_dir = output_base / "img2img"
    for d in (t2i_dir, inpaint_dir, img2img_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(source_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            src_path = item.get("file_path")
            if not src_path or not Path(src_path).exists():
                continue

            manip = item.get("manipulation_type", "")
            p = Path(src_path)
            if manip == "sd_text2img":
                dest = t2i_dir / p.name
                counts["text2image"] += 1
            elif manip == "sd_inpaint":
                dest = inpaint_dir / p.name
                counts["inpainting"] += 1
            else:
                dest = img2img_dir / p.name
                counts["img2img"] += 1

            if p.resolve() != dest.resolve():
                shutil.copy2(p, dest)
            counts["total"] += 1

    logger.info("Organized synthetic faces: %s", counts)
    return counts


def build_benchmark_splits_from_config(
    cfg: Dict[str, Any],
) -> None:
    """Build standardized D0, D1, D2 benchmark stages driven completely by config."""
    paths_cfg = cfg.get("paths", {})
    raw_path = Path(paths_cfg.get("raw_manifest", "data/raw/manifest.jsonl"))
    curated_path = Path(paths_cfg.get("curated_manifest", "data/synthetic/curated_manifest.jsonl"))
    hard_path = Path(paths_cfg.get("hard_manifest", "data/hard_samples/hard_samples_manifest.jsonl"))
    benchmarks_dir = Path(paths_cfg.get("benchmarks_dir", "data/benchmarks"))
    seed = int(cfg.get("seed", 42))

    seed_everything(seed)

    max_real = cfg.get("generation", {}).get("max_real_samples")
    if max_real is not None:
        max_real = int(max_real)

    # 1. Read real faces
    candidate_records: List[ImageMetadata] = []
    if raw_path.is_file():
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                candidate_records.append(ImageMetadata.from_dict(json.loads(line)))
                if max_real is not None and len(candidate_records) >= max_real:
                    break

    # 2. Read curated synthetic faces (excluding duplicates that were marked hard)
    hard_ids = set()
    if hard_path.is_file():
        with open(hard_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = ImageMetadata.from_dict(json.loads(line))
                    candidate_records.append(m)
                    hard_ids.add(m.image_id)

    if curated_path.is_file():
        with open(curated_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    m = ImageMetadata.from_dict(json.loads(line))
                    if m.image_id not in hard_ids:
                        candidate_records.append(m)

    # 3. Assign identity splits ONCE across all candidate records
    ratios = cfg.get("benchmarks", {}).get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    assignment = assign_identity_splits(candidate_records, ratios=ratios, seed=seed)

    # 4. Build each benchmark stage
    stages = cfg.get("benchmarks", {}).get("stages", [])
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    for stage_spec in stages:
        stage_name = stage_spec["name"]
        stage_dir = benchmarks_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        splits = build_stage(candidate_records, assignment=assignment, stage_spec=stage_spec)

        for split_name, metas in splits.items():
            out_file = stage_dir / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as sf:
                for m in metas:
                    sf.write(json.dumps(m.to_dict()) + "\n")

        logger.info(
            "Saved %s splits to %s: %d train, %d val, %d test (identity-disjoint).",
            stage_name,
            stage_dir,
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize dataset into Topic 11 standardized structure.")
    parser.add_argument("--config", type=str, default="configs/dataset_factory.yaml", help="Path to YAML config")
    parser.add_argument("--raw-manifest", type=str, default=None)
    parser.add_argument("--synthetic-manifest", type=str, default=None)
    parser.add_argument("--curated-manifest", type=str, default=None)
    parser.add_argument("--hard-manifest", type=str, default=None)
    parser.add_argument("--base-dir", type=str, default="data")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {}
    if Path(args.config).is_file():
        cfg = load_config(Path(args.config))

    syn_m = Path(resolve(args.synthetic_manifest, cfg, "paths.fake_manifest"))
    cur_m = Path(resolve(args.curated_manifest, cfg, "paths.curated_manifest"))
    hard_m = Path(resolve(args.hard_manifest, cfg, "paths.hard_manifest"))

    base = Path(args.base_dir)
    logger.info("Beginning dataset reorganization in %s...", base)

    # 1. Organize synthetic faces by mode into data/synthetic/
    if syn_m.is_file():
        organize_synthetic_faces(
            source_manifest=syn_m,
            output_base=base / "synthetic",
        )

    # 2. Build benchmark splits D0, D1, D2 in data/benchmarks/
    build_benchmark_splits_from_config(cfg)

    # 3. Export auto-generated dataset statistics
    compute_dataset_stats(cfg)

    logger.info("Dataset reorganization and stats generation complete!")


if __name__ == "__main__":
    main()
