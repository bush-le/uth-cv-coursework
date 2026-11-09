#!/usr/bin/env python3
"""CLI entry point to execute the Reusable EDA Pipeline on any project manifest.

Usage examples:
    python scripts/run_eda.py --manifest data/raw/manifest.jsonl
    python scripts/run_eda.py --manifest data/samples/sample_manifest.jsonl --output-dir experiments/eda_samples
    python scripts/run_eda.py --manifest data/raw/manifest.jsonl --max-quality-samples 500 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend for CLI execution
import matplotlib.pyplot as plt

from src.analysis.eda import (
    analyze_aspect_ratio,
    analyze_demographics,
    analyze_image_dimensions,
    analyze_image_quality,
    analyze_resolution,
    check_corrupted_images,
    compare_dataset_sizes,
    compare_datasets,
    detect_duplicate_images,
    display_image_gallery,
    export_eda_results,
    generate_eda_summary,
    get_dataset_summary,
    load_manifest,
    plot_dataset_comparison,
    plot_dataset_distribution,
    plot_demographics,
    plot_image_properties,
    plot_image_quality,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Reusable Computer Vision EDA on a JSONL manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/raw/manifest.jsonl"),
        help="Path to input JSONL manifest (default: data/raw/manifest.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/eda"),
        help="Directory to store figures, tables, reports, and quality_audit.",
    )
    parser.add_argument(
        "--max-quality-samples",
        type=int,
        default=500,
        help="Maximum images per dataset group to sample for pixel quality metrics (default: 500).",
    )
    parser.add_argument(
        "--check-integrity",
        action="store_true",
        help="Run full byte-level corruption and perceptual duplicate audit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling and galleries (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting Reusable EDA Pipeline on manifest: %s", args.manifest)

    df = load_manifest(args.manifest, project_root=REPO_ROOT, only_existing=False)
    if df.empty:
        logger.warning("Manifest is empty. Exiting EDA pipeline.")
        return

    summary_df = get_dataset_summary(df)
    size_comp_df = compare_dataset_sizes(df)
    demo_res = analyze_demographics(df, only_existing=True)

    dim_df = analyze_image_dimensions(df, random_state=args.seed, show_progress=True)
    res_df = analyze_resolution(dim_df)
    ar_df = analyze_aspect_ratio(dim_df)

    qual_res = analyze_image_quality(
        df,
        max_samples_per_group=args.max_quality_samples,
        random_state=args.seed,
        show_progress=True,
    )
    qual_per_img = qual_res.get("per_image", None)
    qual_sum_df = qual_res.get("summary", None)

    comp_df = compare_datasets(df, dim_df=dim_df, quality_df=qual_per_img)

    audit_tables = {}
    dups = None
    if args.check_integrity:
        integrity_df = check_corrupted_images(df, show_progress=True)
        dups = detect_duplicate_images(df, max_samples=args.max_quality_samples, random_state=args.seed)
        audit_tables["image_integrity_report"] = integrity_df
        audit_tables["exact_duplicates"] = dups["exact_duplicates"]
        audit_tables["perceptual_duplicates"] = dups["perceptual_duplicates"]

    if qual_per_img is not None and not qual_per_img.empty:
        audit_tables["image_quality_per_sample"] = qual_per_img

    # Generate Visualizations
    figures = {
        "01_dataset_distribution": plot_dataset_distribution(summary_df),
        "02_demographics_panel": plot_demographics(demo_res),
        "03_image_properties": plot_image_properties(dim_df),
        "04_image_quality_panel": plot_image_quality(qual_per_img) if qual_per_img is not None else None,
        "05_dataset_comparison_bias": plot_dataset_comparison(qual_per_img) if qual_per_img is not None else None,
        "06_representative_gallery": display_image_gallery(df, n_per_group=4, random_state=args.seed),
    }

    tables = {
        "dataset_summary": summary_df,
        "dataset_size_comparison": size_comp_df,
        "resolution_summary": res_df,
        "aspect_ratio_summary": ar_df,
        "image_quality_summary": qual_sum_df,
        "cross_dataset_comparison": comp_df,
        **{f"demo_{k}": v for k, v in demo_res.items()},
    }

    report_md = generate_eda_summary(
        summary_df=summary_df,
        demo_results=demo_res,
        res_df=res_df,
        quality_summary_df=qual_sum_df,
        comparison_df=comp_df,
        duplicates_dict=dups,
    )

    exported = export_eda_results(
        output_dir=REPO_ROOT / args.output_dir,
        tables=tables,
        figures=figures,
        report_markdown=report_md,
        quality_audit_tables=audit_tables,
    )
    plt.close("all")
    logger.info("EDA completed successfully. %d artifacts written to %s", len(exported), args.output_dir)


if __name__ == "__main__":
    main()