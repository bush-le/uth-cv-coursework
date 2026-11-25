"""Full Dual-Engine Portfolio Extended Quality Audit Scanner (6,545 SD 1.5 + 6,000 RV5.1).

Audits all existing raw/candidate synthetic image pools on disk without modifying
production manifests or checkpoints, outputting a consolidated report at
`experiments/logs/full_portfolio_extended_audit.json`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.logger import get_logger
from src.pipelines.curate_image_dataset import (
    ImageCurationGate,
    calibrate_sharpness_from_manifest,
    curate_image_manifest,
)
from src.pipelines.run_end_to_end_showcase import _aggregate_quality_stats, _compute_diversity_metrics, _read_jsonl

logger = get_logger("scan_full_portfolio_audit")

PORTFOLIO_POOLS = [
    ("sd15_block1_raw", Path("data/synthetic/fake_manifest.jsonl")),
    ("sd15_al_round1", Path("data/active_learning/round_1/candidates_raw.jsonl")),
    ("sd15_al_round2", Path("data/active_learning/round_2/candidates_raw.jsonl")),
    ("rv51_ood_6k", Path("data/synthetic/realimage_v51_6k/fake_manifest.jsonl")),
]


def _scan_one_pool(
    args: Tuple[str, Path, Path, Path, float, Optional[int]],
) -> Tuple[str, Dict[str, Any]]:
    pool_name, in_manifest, output_root, raw_manifest, min_sharp, perceptual_dedup_threshold = args
    pool_dir = output_root / pool_name
    out_cur = pool_dir / "curated_manifest.jsonl"
    out_rep = pool_dir / "curation_report.jsonl"

    gate = ImageCurationGate(
        min_sharpness=min_sharp,
        min_clip_score=0.20,
        clip_enabled=True,
        device="auto",
        face_backend="yunet",
        dedup=True,
        perceptual_dedup_threshold=perceptual_dedup_threshold,
        extended_audit=True,
    )
    logger.info("Scanning pool [%s] (%s)...", pool_name, in_manifest)
    cur_sum = curate_image_manifest(
        input_manifest=in_manifest,
        output_manifest=out_cur,
        gate=gate,
        curation_report=out_rep,
        raw_manifest=raw_manifest,
        source_dirs=[Path("data/raw/faces/celeba_hq"), Path("data/raw/faces/ffhq")],
    )

    raw_items = _read_jsonl(in_manifest)
    cur_items = _read_jsonl(out_cur)
    rep_items = _read_jsonl(out_rep)
    q_stats = _aggregate_quality_stats(raw_items, rep_items)
    div = _compute_diversity_metrics(cur_items, max_dhash_samples=80)

    res = {
        "manifest": str(in_manifest),
        "total": cur_sum["total"],
        "kept": cur_sum["kept"],
        "rejected": cur_sum["rejected"],
        "pass_rate": cur_sum["kept"] / max(1, cur_sum["total"]),
        "diversity": div,
        "quality_stats": q_stats,
    }
    logger.info(
        "Pool [%s] complete: %d/%d passed (%.2f%%)",
        pool_name,
        cur_sum["kept"],
        cur_sum["total"],
        100.0 * res["pass_rate"],
    )
    return pool_name, res


def scan_portfolio(
    output_root: Path = Path("experiments/demo_runs/full_audit"),
    summary_out: Path = Path("experiments/logs/full_portfolio_extended_audit.json"),
    raw_manifest: Path = Path("data/raw/manifest.jsonl"),
    perceptual_dedup_threshold: Optional[int] = 2,
    pools: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Scan all 4 raw synthetic pools (or a named subset) in parallel with extended audit."""
    output_root.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    min_sharp = calibrate_sharpness_from_manifest(raw_manifest, ref_quantile=0.05, max_samples=50, fallback=1.25)
    logger.info("Calibrated min_sharpness=%.2f from %s", min_sharp, raw_manifest)

    target_set = set(pools) if pools else None
    tasks = [
        (pool_name, in_manifest, output_root, raw_manifest, min_sharp, perceptual_dedup_threshold)
        for pool_name, in_manifest in PORTFOLIO_POOLS
        if in_manifest.is_file() and (target_set is None or pool_name in target_set)
    ]

    pool_results: Dict[str, Any] = {}
    if target_set is not None and summary_out.is_file():
        existing = json.loads(summary_out.read_text(encoding="utf-8"))
        pool_results.update(existing.get("pools", {}))

    # Cap workers at 2 so concurrent CLIP ViT-B/32 CUDA instances stay comfortably within 4GB VRAM
    with ProcessPoolExecutor(max_workers=min(2, max(1, len(tasks)))) as ex:
        futures = [ex.submit(_scan_one_pool, t) for t in tasks]
        for fut in as_completed(futures):
            p_name, p_res = fut.result()
            pool_results[p_name] = p_res

    ordered_results = {name: pool_results[name] for name, _ in PORTFOLIO_POOLS if name in pool_results}
    total_scanned = sum(v["total"] for v in ordered_results.values())
    total_passed = sum(v["kept"] for v in ordered_results.values())

    full_report: Dict[str, Any] = {
        "min_sharpness_threshold": min_sharp,
        "perceptual_dedup_threshold": perceptual_dedup_threshold,
        "total_scanned": total_scanned,
        "total_passed": total_passed,
        "total_rejected": total_scanned - total_passed,
        "overall_pass_rate": total_passed / max(1, total_scanned),
        "pools": ordered_results,
    }
    summary_out.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved full portfolio extended audit -> %s", summary_out)
    return full_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan full SD1.5 + RV5.1 portfolio with extended quality gate.")
    parser.add_argument("--output-root", type=str, default="experiments/demo_runs/full_audit")
    parser.add_argument("--summary-out", type=str, default="experiments/logs/full_portfolio_extended_audit.json")
    parser.add_argument("--dhash-thresh", type=int, default=2, help="Perceptual dHash threshold (<0 disables).")
    parser.add_argument("--pools", nargs="*", default=None, help="Optional subset of pool names to re-scan.")
    args = parser.parse_args()

    scan_portfolio(
        output_root=Path(args.output_root),
        summary_out=Path(args.summary_out),
        perceptual_dedup_threshold=args.dhash_thresh if args.dhash_thresh >= 0 else None,
        pools=args.pools,
    )


if __name__ == "__main__":
    main()
