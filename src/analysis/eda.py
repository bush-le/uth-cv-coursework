"""Reusable Exploratory Data Analysis (EDA) module for 2D facial & deepfake datasets.

Supports multi-stage manifests:
1. Raw Real Faces (FFHQ, CelebA-HQ)
2. Processed Datasets
3. Synthetic Stable Diffusion Datasets (Text2Img, Inpaint, Img2Img)
4. Hard Samples (Bypassed DINOv2)
5. Benchmark Splits (D0, D1, D2)

Design Principles:
- Strict separation between Computation (DataFrames/Dicts), Visualization (Matplotlib Figures),
  and Export (CSV/Markdown/PNG artifacts).
- Zero hard-coded paths; automatic project-root resolution so functions work identically
  from CLI scripts, pytest, or nested Jupyter Notebooks.
- Memory-safe streaming image inspection (never loads full dataset into RAM).
- Read-only guarantee on original image files.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.common.logger import get_logger
from src.data.image_schema import ImageMetadata

logger = get_logger(__name__)

PathLike = Union[str, Path]


# ============================================================================
# 0. PATH & SCHEMA HELPERS
# ============================================================================

def resolve_project_root(start_path: Optional[PathLike] = None) -> Path:
    """Resolve the repository root directory dynamically from any working directory.

    Args:
        start_path: Optional starting directory or file path.

    Returns:
        Resolved Path pointing to the repository root.
    """
    curr = Path(start_path or Path.cwd()).resolve()
    if curr.is_file():
        curr = curr.parent

    for candidate in [curr] + list(curr.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / "src").is_dir():
            return candidate
    return curr


def _resolve_image_path(file_path: Optional[str], project_root: Path) -> Optional[Path]:
    """Resolve relative or absolute image path safely against project_root."""
    if not file_path:
        return None
    p = Path(file_path)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()

def _parse_bool(val: Any) -> Optional[bool]:
    """Ép kiểu an toàn cho các dạng True/False/1/0/'Yes'/'No'."""
    if val is None: return None
    if isinstance(val, bool): return val
    if isinstance(val, (int, float)):
        if val == 1: return True
        if val == 0: return False
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"true", "1", "yes", "t", "y"}: return True
        if v in {"false", "0", "no", "none", "f", "n"}: return False
    return None

def _harmonize_demographics(demo: Optional[Dict[str, Any]], fallback_source: str) -> Dict[str, Any]:
    """Normalize heterogeneous demographic schemas (FFHQ vs CelebA-HQ vs Synthetic) without fabrication.

    Distinguishes explicitly between:
    - False / 'without_glasses' (confirmed negative attribute)
    - 'unknown' (metadata record exists but field is unannotated)
    - 'not_applicable' / 'missing' (no demographic dict provided)
    """
    if not demo or not isinstance(demo, dict):
        return {
            "dataset": fallback_source or "unknown",
            "gender_norm": "missing",
            "age_group_raw": "missing",
            "young_norm": "missing",
            "glasses_norm": "missing",
            "glasses_detail": "missing",
            "identity_id": "missing",
            "smiling": "missing",
            "yaw": np.nan,
            "pitch": np.nan,
            "roll": np.nan,
        }

    dataset_name = str(demo.get("dataset", fallback_source or "unknown")).lower()

    # 1. Gender harmonization ('male' / 'female' / 'unknown')
    if "gender" in demo and demo["gender"] is not None:
        g_raw = str(demo["gender"]).strip().lower()
        gender_norm = g_raw if g_raw in {"male", "female"} else "unknown"
    elif "male" in demo and demo["male"] is not None:
        gender_norm = "male" if _parse_bool(demo["male"]) else "female"
    else:
        gender_norm = "unknown"

    # 2. Age harmonization
    age_group_raw = str(demo.get("age_group", "not_applicable" if "young" in demo else "unknown"))
    if "young" in demo and demo["young"] is not None:
        young_norm = "young" if _parse_bool(demo["young"]) else "non_young"
    elif age_group_raw not in {"unknown", "not_applicable", "missing"}:
        young_buckets = {"0-2", "3-6", "7-9", "10-14", "15-19", "20-29", "30-39"}
        young_norm = "young" if age_group_raw in young_buckets else "non_young"
    else:
        young_norm = "unknown"

    # 3. Glasses / Occlusion harmonization
    if "glasses" in demo and demo["glasses"] is not None:
        gl_raw = str(demo["glasses"]).strip()
        glasses_detail = gl_raw
        glasses_norm = "without_glasses" if gl_raw.lower() in {"none", "false", "0", "no"} else "with_glasses"
    elif "eyeglasses" in demo and demo["eyeglasses"] is not None:
        has_gl = _parse_bool(demo["eyeglasses"])
        glasses_detail = "Eyeglasses" if has_gl else "None"
        glasses_norm = "with_glasses" if has_gl else "without_glasses"
    else:
        glasses_detail = "unknown"
        glasses_norm = "unknown"

    # 4. Identity & additional attributes
    identity_id = str(demo.get("identity_id", "unknown"))
    smiling = (
        ("smiling" if _parse_bool(demo["smiling"]) else "not_smiling")
        if "smiling" in demo and demo["smiling"] is not None
        else "unknown"
    )

    # 5. Head pose angles
    def _to_float(val: Any) -> float:
        try:
            return float(val) if val is not None else np.nan
        except (ValueError, TypeError):
            return np.nan

    yaw = _to_float(demo.get("yaw", demo.get("head_yaw")))
    pitch = _to_float(demo.get("pitch", demo.get("head_pitch")))
    roll = _to_float(demo.get("roll", demo.get("head_roll")))

    return {
        "dataset": dataset_name,
        "gender_norm": gender_norm,
        "age_group_raw": age_group_raw,
        "young_norm": young_norm,
        "glasses_norm": glasses_norm,
        "glasses_detail": glasses_detail,
        "identity_id": identity_id,
        "smiling": smiling,
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
    }


# ============================================================================
# A. DATASET OVERVIEW
# ============================================================================

def load_manifest(
    manifest_path: PathLike,
    project_root: Optional[PathLike] = None,
    only_existing: bool = False,
) -> pd.DataFrame:
    """Load a JSONL manifest into a standardized, analysis-ready Pandas DataFrame.

    Resolves relative image paths against the repository root automatically and checks
    physical file existence and byte size on disk without loading images into RAM.

    Args:
        manifest_path: Path to the .jsonl manifest file.
        project_root: Optional repository root path. Auto-detected if None.
        only_existing: If True, filter the returned DataFrame to only existing files on disk.

    Returns:
        DataFrame with standardized metadata, resolved paths, file existence flags,
        and harmonized demographic columns.
    """
    root = resolve_project_root(project_root)
    m_path = Path(manifest_path)
    if not m_path.is_absolute():
        candidate = (root / m_path).resolve()
        if candidate.exists():
            m_path = candidate

    if not m_path.exists():
        raise FileNotFoundError(
            f"Manifest file not found at '{manifest_path}' (resolved: '{m_path}'). "
            f"Please check dataset setup or run 'python scripts/build_raw_dataset.py'."
        )

    rows: List[Dict[str, Any]] = []
    malformed_lines = 0

    with open(m_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw_dict = json.loads(stripped)
                meta = ImageMetadata.from_dict(raw_dict)
            except Exception as exc:
                malformed_lines += 1
                logger.warning("Skipping malformed JSONL line %d in %s: %s", line_idx, m_path, exc)
                continue

            resolved_p = _resolve_image_path(meta.file_path, root)
            file_exists = bool(resolved_p and resolved_p.is_file())
            file_size_bytes = resolved_p.stat().st_size if file_exists and resolved_p else 0
            extension = resolved_p.suffix.lower() if resolved_p else ""
            filename = resolved_p.name if resolved_p else ""

            demo_norm = _harmonize_demographics(meta.demographics, fallback_source=meta.data_source)

            row = {
                "image_id": meta.image_id,
                "label": meta.label,
                "label_name": "real" if meta.label == 0 else "fake",
                "manipulation_type": meta.manipulation_type or "unknown",
                "prompt": meta.prompt,
                "source_real_id": meta.source_real_id,
                "file_path": meta.file_path,
                "resolved_path": str(resolved_p) if resolved_p else None,
                "filename": filename,
                "extension": extension,
                "file_exists": file_exists,
                "file_size_kb": round(file_size_bytes / 1024.0, 2),
                "bbox": meta.bbox,
                "is_hard_sample": meta.is_hard_sample,
                "baseline_score": meta.baseline_score,
                "data_source": meta.data_source,
                **demo_norm,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("Manifest at %s produced an empty DataFrame (malformed=%d).", m_path, malformed_lines)
        return df

    total = len(df)
    existing = int(df["file_exists"].sum())
    missing = total - existing
    logger.info(
        "Loaded manifest '%s': total=%d, existing_on_disk=%d, missing_on_disk=%d",
        m_path.name,
        total,
        existing,
        missing,
    )

    if only_existing:
        df = df[df["file_exists"]].copy().reset_index(drop=True)
        logger.info("Filtered to %d existing image records.", len(df))

    return df


def get_dataset_summary(
    df: pd.DataFrame,
    group_col: str = "dataset",
) -> pd.DataFrame:
    """Compute high-level inventory, file availability, extensions, and storage statistics.

    Args:
        df: Manifest DataFrame returned by load_manifest().
        group_col: Column to group by (default: 'dataset', fallback: 'data_source').

    Returns:
        Summary DataFrame with one row per dataset source.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "manifest_records",
                "existing_files",
                "missing_files",
                "availability_pct",
                "duplicate_filenames",
                "extensions",
                "total_storage_mb",
                "mean_file_size_kb",
                "real_count",
                "fake_count",
                "hard_sample_count",
            ]
        )

    col = group_col if group_col in df.columns else "data_source"
    summaries: List[Dict[str, Any]] = []

    for grp, sub in df.groupby(col, dropna=False):
        n_total = len(sub)
        existing_sub = sub[sub["file_exists"]]
        n_exist = len(existing_sub)
        n_missing = n_total - n_exist
        avail_pct = round((n_exist / n_total) * 100.0, 2) if n_total > 0 else 0.0

        dup_names = int(existing_sub["filename"].duplicated().sum()) if not existing_sub.empty else 0
        exts = ", ".join(sorted(existing_sub["extension"].dropna().unique())) if not existing_sub.empty else "N/A"
        total_mb = round(existing_sub["file_size_kb"].sum() / 1024.0, 2) if not existing_sub.empty else 0.0
        mean_kb = round(existing_sub["file_size_kb"].mean(), 2) if not existing_sub.empty else 0.0

        real_cnt = int((sub["label"] == 0).sum()) if "label" in sub.columns else 0
        fake_cnt = int((sub["label"] == 1).sum()) if "label" in sub.columns else 0
        hard_cnt = int(sub["is_hard_sample"].sum()) if "is_hard_sample" in sub.columns else 0

        summaries.append(
            {
                "group": str(grp),
                "manifest_records": n_total,
                "existing_files": n_exist,
                "missing_files": n_missing,
                "availability_pct": avail_pct,
                "duplicate_filenames": dup_names,
                "extensions": exts,
                "total_storage_mb": total_mb,
                "mean_file_size_kb": mean_kb,
                "real_count": real_cnt,
                "fake_count": fake_cnt,
                "hard_sample_count": hard_cnt,
            }
        )

    return pd.DataFrame(summaries).sort_values("manifest_records", ascending=False).reset_index(drop=True)


def compare_dataset_sizes(
    df: pd.DataFrame,
    group_col: str = "dataset",
) -> pd.DataFrame:
    """Compare planned/manifest counts vs. available disk counts and relative proportions."""
    summary = get_dataset_summary(df, group_col=group_col)
    if summary.empty:
        return summary

    total_manifest = summary["manifest_records"].sum()
    total_existing = summary["existing_files"].sum()

    summary["manifest_share_pct"] = (
        (summary["manifest_records"] / total_manifest * 100.0).round(2) if total_manifest > 0 else 0.0
    )
    summary["existing_share_pct"] = (
        (summary["existing_files"] / total_existing * 100.0).round(2) if total_existing > 0 else 0.0
    )
    return summary[
        [
            "group",
            "manifest_records",
            "manifest_share_pct",
            "existing_files",
            "existing_share_pct",
            "missing_files",
            "availability_pct",
            "total_storage_mb",
        ]
    ]


# ============================================================================
# B. DEMOGRAPHIC ANALYSIS
# ============================================================================

def _compute_distribution_table(
    df: pd.DataFrame,
    attr_col: str,
    group_col: str = "dataset",
) -> pd.DataFrame:
    """Helper to compute counts and within-group percentages for a demographic attribute."""
    if df.empty or attr_col not in df.columns:
        logger.warning("Column '%s' not available for demographic analysis.", attr_col)
        return pd.DataFrame(columns=[group_col, attr_col, "count", "percentage"])

    g_col = group_col if group_col in df.columns else "data_source"
    counts = (
        df.groupby([g_col, attr_col], dropna=False)
        .size()
        .reset_index(name="count")
    )
    group_totals = counts.groupby(g_col)["count"].transform("sum")
    counts["percentage"] = ((counts["count"] / group_totals) * 100.0).round(2)
    return counts.sort_values([g_col, "count"], ascending=[True, False]).reset_index(drop=True)


def analyze_age_distribution(
    df: pd.DataFrame,
    only_existing: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Analyze both granular age groups (e.g. FFHQ) and harmonized binary young/non_young status.

    Never fabricates granular age buckets for datasets that only provide binary 'Young' flags.
    """
    work_df = df[df["file_exists"]].copy() if (only_existing and "file_exists" in df.columns) else df.copy()
    return {
        "granular_age_group": _compute_distribution_table(work_df, "age_group_raw"),
        "harmonized_young_status": _compute_distribution_table(work_df, "young_norm"),
    }


def analyze_gender_distribution(
    df: pd.DataFrame,
    only_existing: bool = True,
) -> pd.DataFrame:
    """Analyze harmonized gender distribution ('male', 'female', 'unknown', 'missing') per dataset."""
    work_df = df[df["file_exists"]].copy() if (only_existing and "file_exists" in df.columns) else df.copy()
    return _compute_distribution_table(work_df, "gender_norm")


def analyze_glasses_distribution(
    df: pd.DataFrame,
    only_existing: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Analyze both harmonized glasses presence and dataset-specific glasses types (None/Normal/Dark)."""
    work_df = df[df["file_exists"]].copy() if (only_existing and "file_exists" in df.columns) else df.copy()
    return {
        "harmonized_glasses": _compute_distribution_table(work_df, "glasses_norm"),
        "detailed_glasses": _compute_distribution_table(work_df, "glasses_detail"),
    }


def analyze_demographics(
    df: pd.DataFrame,
    only_existing: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Comprehensive demographic analysis including Age, Gender, Glasses, Identity, and Head Pose."""
    work_df = df[df["file_exists"]].copy() if (only_existing and "file_exists" in df.columns) else df.copy()
    if work_df.empty:
        logger.warning("Empty DataFrame passed to analyze_demographics().")
        return {}

    age_res = analyze_age_distribution(work_df, only_existing=False)
    glasses_res = analyze_glasses_distribution(work_df, only_existing=False)
    gender_df = analyze_gender_distribution(work_df, only_existing=False)

    # Head pose summary (for datasets providing yaw, pitch, roll)
    pose_rows: List[Dict[str, Any]] = []
    if {"yaw", "pitch", "roll"}.issubset(work_df.columns):
        for grp, sub in work_df.groupby("dataset", dropna=False):
            valid_pose = sub[["yaw", "pitch", "roll"]].dropna(how="all")
            if not valid_pose.empty:
                for angle in ["yaw", "pitch", "roll"]:
                    s = valid_pose[angle].dropna()
                    if not s.empty:
                        pose_rows.append(
                            {
                                "dataset": str(grp),
                                "angle": angle,
                                "count": len(s),
                                "mean_deg": round(float(s.mean()), 2),
                                "std_deg": round(float(s.std()), 2),
                                "min_deg": round(float(s.min()), 2),
                                "median_deg": round(float(s.median()), 2),
                                "max_deg": round(float(s.max()), 2),
                                "extreme_pose_gt15_pct": round(float((s.abs() > 15.0).mean() * 100.0), 2),
                            }
                        )
    pose_df = pd.DataFrame(pose_rows)

    # Identity uniqueness & repetition check (crucial for Identity Leakage prevention)
    identity_rows: List[Dict[str, Any]] = []
    for grp, sub in work_df.groupby("dataset", dropna=False):
        n_samples = len(sub)
        if "source_real_id" in sub.columns and sub["source_real_id"].notna().any():
            valid_ids = sub[sub["source_real_id"] != "unknown"]["source_real_id"].dropna()
            if not valid_ids.empty:
               uniq_ids = valid_ids.nunique()
               top_repeat = int(valid_ids.value_counts().max())
            else:
               uniq_ids = np.nan
               top_repeat = np.nan
        else:
            uniq_ids = np.nan
            top_repeat = np.nan
        identity_rows.append(
            {
                "dataset": str(grp),
                "total_images": n_samples,
                "unique_identities": uniq_ids,
                "avg_images_per_identity": round(n_samples / uniq_ids, 2) if uniq_ids > 0 else 0.0,
                "max_images_single_identity": top_repeat,
            }
        )
    identity_df = pd.DataFrame(identity_rows)

    return {
        "gender": gender_df,
        "age_granular": age_res["granular_age_group"],
        "age_harmonized": age_res["harmonized_young_status"],
        "glasses_harmonized": glasses_res["harmonized_glasses"],
        "glasses_detailed": glasses_res["detailed_glasses"],
        "head_pose": pose_df,
        "identity_summary": identity_df,
    }


# ============================================================================
# C. IMAGE PROPERTIES (DIMENSIONS, RESOLUTION, ASPECT RATIO)
# ============================================================================

def analyze_image_dimensions(
    df: pd.DataFrame,
    max_samples: Optional[int] = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Inspect image headers (width, height, mode, channels, aspect ratio) without decoding full pixels.

    Args:
        df: Manifest DataFrame from load_manifest().
        max_samples: Optional maximum number of existing images to inspect per dataset.
        random_state: Random seed for reproducible subsampling.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        DataFrame with image dimension records per valid image.
    """
    if df.empty or "file_exists" not in df.columns:
        return pd.DataFrame(
            columns=[
                "image_id",
                "dataset",
                "data_source",
                "label",
                "manipulation_type",
                "width",
                "height",
                "resolution_str",
                "megapixels",
                "aspect_ratio",
                "mode",
                "file_size_kb",
            ]
        )

    existing_df = df[df["file_exists"]].copy()
    if existing_df.empty:
        logger.warning("No existing image files on disk to analyze dimensions.")
        return pd.DataFrame()

    if max_samples is not None:
       g_col = "dataset" if "dataset" in existing_df.columns else "data_source"
       existing_df = pd.concat([
           sub.sample(n=min(len(sub), max_samples), random_state=random_state) 
           for _, sub in existing_df.groupby(g_col, dropna=False)
        ], ignore_index=True)
    
    dim_records: List[Dict[str, Any]] = []
    iterator = tqdm(
        existing_df.itertuples(index=False),
        total=len(existing_df),
        desc="Inspecting image dimensions",
        disable=not show_progress,
    )

    for row in iterator:
        p = row.resolved_path
        if not p:
            continue
        try:
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
            dim_records.append(
                {
                    "image_id": row.image_id,
                    "dataset": getattr(row, "dataset", "unknown"),
                    "data_source": getattr(row, "data_source", "baseline"),
                    "label": getattr(row, "label", 0),
                    "manipulation_type": getattr(row, "manipulation_type", "real"),
                    "width": int(w),
                    "height": int(h),
                    "resolution_str": f"{w}x{h}",
                    "megapixels": round((w * h) / 1_000_000.0, 4),
                    "aspect_ratio": round(w / h, 4) if h > 0 else np.nan,
                    "mode": mode,
                    "file_size_kb": getattr(row, "file_size_kb", 0.0),
                }
            )
        except Exception as exc:
            logger.warning("Could not read image header for %s: %s", p, exc)

    return pd.DataFrame(dim_records)


def analyze_resolution(
    dim_df: pd.DataFrame,
    group_col: str = "dataset",
) -> pd.DataFrame:
    """Summarize width, height, and resolution groups in a clean, presentation-ready DataFrame."""
    if dim_df.empty:
        return pd.DataFrame()

    g_col = group_col if group_col in dim_df.columns else "data_source"
    rows: List[Dict[str, Any]] = []

    for grp, sub in dim_df.groupby(g_col, dropna=False):
        dominant_res = sub["resolution_str"].mode().iloc[0] if not sub["resolution_str"].empty else "N/A"
        unique_res_cnt = sub["resolution_str"].nunique()
        rows.append(
            {
                "group": str(grp),
                "image_count": len(sub),
                "dominant_resolution": dominant_res,
                "unique_resolutions": unique_res_cnt,
                "mean_width": round(float(sub["width"].mean()), 1),
                "min_width": int(sub["width"].min()),
                "max_width": int(sub["width"].max()),
                "mean_height": round(float(sub["height"].mean()), 1),
                "min_height": int(sub["height"].min()),
                "max_height": int(sub["height"].max()),
                "mean_megapixels": round(float(sub["megapixels"].mean()), 3),
                "mean_aspect_ratio": round(float(sub["aspect_ratio"].mean()), 3),
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def analyze_aspect_ratio(
    dim_df: pd.DataFrame,
    group_col: str = "dataset",
    tolerance: float = 0.01,
) -> pd.DataFrame:
    """Analyze aspect ratio distribution and flag non-square geometric anomalies."""
    if dim_df.empty:
        return pd.DataFrame()

    g_col = group_col if group_col in dim_df.columns else "data_source"
    rows: List[Dict[str, Any]] = []

    for grp, sub in dim_df.groupby(g_col, dropna=False):
        ar = sub["aspect_ratio"].dropna()
        non_square = int((np.abs(ar - 1.0) > tolerance).sum())
        rows.append(
            {
                "group": str(grp),
                "image_count": len(ar),
                "mean_ar": round(float(ar.mean()), 4),
                "std_ar": round(float(ar.std()), 4),
                "min_ar": round(float(ar.min()), 4),
                "median_ar": round(float(ar.median()), 4),
                "max_ar": round(float(ar.max()), 4),
                "non_square_count": non_square,
                "square_Numeric_pct": round(((len(ar) - non_square) / len(ar)) * 100.0, 2) if len(ar) > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================================
# D. IMAGE QUALITY & INTEGRITY METRICS
# ============================================================================

def calculate_brightness(rgb_or_gray: np.ndarray) -> float:
    """Calculate Mean Pixel Intensity in standard ITU-R BT.601 luminance [0, 255]."""
    arr = np.asarray(rgb_or_gray, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr
    return float(np.mean(gray))


def calculate_contrast(rgb_or_gray: np.ndarray) -> float:
    """Calculate RMS Contrast (Standard Deviation of luminance pixel intensities)."""
    arr = np.asarray(rgb_or_gray, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr
    return float(np.std(gray))


def calculate_sharpness(rgb_or_gray: np.ndarray) -> float:
    """Calculate image sharpness using the Variance of the discrete 2D Laplacian operator.

    Applies the standard 3x3 discrete Laplacian kernel:
        [[ 0,  1,  0],
         [ 1, -4,  1],
         [ 0,  1,  0]]
    Higher variance indicates stronger high-frequency edge responses (sharper image).
    """
    arr = np.asarray(rgb_or_gray, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    center = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * center
    )
    return float(np.var(laplacian))


def _compute_dhash(pil_img: Image.Image, hash_size: int = 8) -> str:
    """Compute 64-bit Difference Hash (dHash) in pure NumPy/Pillow without external dependencies."""
    small = pil_img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bit_string = "".join("1" if b else "0" for b in diff.flatten())
    return f"{int(bit_string, 2):016x}"


def check_corrupted_images(
    df: pd.DataFrame,
    min_dimension: int = 32,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Verify byte-level and decode-level integrity of all manifest records.

    Never deletes or modifies any files. Flags missing files, zero-byte files,
    corrupted headers, truncated pixel streams, and abnormally small images.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["image_id", "dataset", "file_path", "status", "is_valid", "width", "height", "error_msg"]
        )

    audit_rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Auditing image integrity",
        disable=not show_progress,
    )

    for row in iterator:
        p_str = getattr(row, "resolved_path", None)
        p = Path(p_str) if p_str else None
        dataset_name = getattr(row, "dataset", "unknown")

        if p is None or not p.exists():
            audit_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": dataset_name,
                    "file_path": row.file_path,
                    "status": "missing_file",
                    "is_valid": False,
                    "width": np.nan,
                    "height": np.nan,
                    "error_msg": "File does not exist on disk",
                }
            )
            continue

        if p.stat().st_size == 0:
            audit_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": dataset_name,
                    "file_path": row.file_path,
                    "status": "empty_file",
                    "is_valid": False,
                    "width": np.nan,
                    "height": np.nan,
                    "error_msg": "Zero-byte file",
                }
            )
            continue

        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im2:
                im2.load()
                w, h = im2.size

            if w < min_dimension or h < min_dimension:
                status = "too_small"
                is_valid = False
                err = f"Image dimensions {w}x{h} below minimum {min_dimension}px"
            else:
                status = "valid"
                is_valid = True
                err = ""

            audit_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": dataset_name,
                    "file_path": row.file_path,
                    "status": status,
                    "is_valid": is_valid,
                    "width": w,
                    "height": h,
                    "error_msg": err,
                }
            )
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            audit_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": dataset_name,
                    "file_path": row.file_path,
                    "status": "corrupted",
                    "is_valid": False,
                    "width": np.nan,
                    "height": np.nan,
                    "error_msg": str(exc),
                }
            )

    return pd.DataFrame(audit_rows)

def _hamming_distance(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")

def detect_duplicate_images(
    df: pd.DataFrame,
    max_samples: Optional[int] = None,
    random_state: int = 42,
    threshold: int = 10,
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Detect exact byte duplicates (MD5) and near-perceptual duplicates (64-bit dHash)."""
    if df.empty or "file_exists" not in df.columns:
        return {"exact_duplicates": pd.DataFrame(), "perceptual_duplicates": pd.DataFrame()}

    existing_df = df[df["file_exists"]].copy()
    if existing_df.empty:
        return {"exact_duplicates": pd.DataFrame(), "perceptual_duplicates": pd.DataFrame()}

    if max_samples is not None and len(existing_df) > max_samples:
        existing_df = existing_df.sample(n=max_samples, random_state=random_state).sort_index()

    hash_rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        existing_df.itertuples(index=False),
        total=len(existing_df),
        desc="Computing MD5 & Perceptual Hashes",
        disable=not show_progress,
    )

    for row in iterator:
        p = Path(row.resolved_path)
        try:
            md5_hash = hashlib.md5()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""): 
                   md5_hash.update(chunk)
            md5_hex = md5_hash.hexdigest()

            with Image.open(p) as im:
                dhash_hex = _compute_dhash(im)
            hash_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": getattr(row, "dataset", "unknown"),
                    "file_path": row.file_path,
                    "md5": md5_hex,
                    "dhash": dhash_hex,
                }
            )
        except Exception as exc:
            logger.warning("Skipping hash calculation for %s: %s", p, exc)

    hash_df = pd.DataFrame(hash_rows)
    if hash_df.empty:
        return {"exact_duplicates": pd.DataFrame(), "perceptual_duplicates": pd.DataFrame()}

    exact_dups = hash_df[hash_df.duplicated(subset=["md5"], keep=False)].sort_values("md5").reset_index(drop=True)
    perc_recs = []
    valid_dhash = hash_df.dropna(subset=["dhash"])
    for (i1, r1), (i2, r2) in combinations(valid_dhash.iterrows(), 2):
         if _hamming_distance(r1["dhash"], r2["dhash"]) <= threshold:
            perc_recs.append({"img1": r1["image_id"], "img2": r2["image_id"]})
    perceptual_dups = pd.DataFrame(perc_recs)

    return {
        "exact_duplicates": exact_dups,
        "perceptual_duplicates": perceptual_dups,
    }


def analyze_image_quality(
    df: pd.DataFrame,
    max_samples_per_group: Optional[int] = 500,
    group_col: str = "dataset",
    random_state: int = 42,
    compute_Downscaled_size: Tuple[int, int] = (256, 256),
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Compute Sharpness, Brightness, Contrast, and RGB Channel statistics per image.

    Uses stratified sampling per dataset group (`max_samples_per_group`) so both FFHQ and
    CelebA-HQ are fairly represented without biases from manifest ordering.
    """
    if df.empty or "file_exists" not in df.columns:
        return {"per_image": pd.DataFrame(), "summary": pd.DataFrame()}

    existing_df = df[df["file_exists"]].copy()
    if existing_df.empty:
        return {"per_image": pd.DataFrame(), "summary": pd.DataFrame()}

    g_col = group_col if group_col in existing_df.columns else "data_source"

    if max_samples_per_group is not None:
        sampled_parts: List[pd.DataFrame] = []
        for _, sub in existing_df.groupby(g_col, dropna=False):
            if len(sub) > max_samples_per_group:
                sampled_parts.append(sub.sample(n=max_samples_per_group, random_state=random_state))
            else:
                sampled_parts.append(sub)
        work_df = pd.concat(sampled_parts, ignore_index=True)
    else:
        work_df = existing_df

    quality_rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        work_df.itertuples(index=False),
        total=len(work_df),
        desc="Analyzing optical & RGB quality",
        disable=not show_progress,
    )

    for row in iterator:
        p = Path(row.resolved_path)
        try:
            with Image.open(p) as im:
                rgb_im = im.convert("RGB")
                arr_orig = np.asarray(rgb_im, dtype=np.float32)
    
    # Bắt buộc tính sharpness trên ảnh gốc, chưa bị resize
                sharpness = calculate_sharpness(arr_orig)
    
                if compute_Downscaled_size is not None:
        # Dùng thumbnail để không làm méo tỷ lệ khung hình
                   rgb_im.thumbnail(compute_Downscaled_size, Image.Resampling.BILINEAR)
                   arr_proc = np.asarray(rgb_im, dtype=np.float32)
                else:
                   arr_proc = arr_orig

            brightness = calculate_brightness(arr_proc) # Sửa arr thành arr_proc cho các dòng dưới
            contrast = calculate_contrast(arr_proc)

            r_mean = float(np.mean(arr_proc[:, :, 0]))
            g_mean = float(np.mean(arr_proc[:, :, 1]))
            b_mean = float(np.mean(arr_proc[:, :, 2]))

            quality_rows.append(
                {
                    "image_id": row.image_id,
                    "dataset": getattr(row, g_col, "unknown"),
                    "label": getattr(row, "label", 0),
                    "manipulation_type": getattr(row, "manipulation_type", "real"),
                    "file_path": row.file_path,
                    "resolved_path": row.resolved_path,
                    "sharpness": round(sharpness, 2),
                    "brightness": round(brightness, 2),
                    "contrast": round(contrast, 2),
                    "red_mean": round(r_mean, 2),
                    "green_mean": round(g_mean, 2),
                    "blue_mean": round(b_mean, 2),
                }
            )
        except Exception as exc:
            logger.warning("Skipping quality metrics for %s: %s", p, exc)

    q_df = pd.DataFrame(quality_rows)
    if q_df.empty:
        return {"per_image": q_df, "summary": pd.DataFrame()}

    # Flag relative percentile outliers within each dataset (5th and 95th percentiles)
    q_df["quality_flag"] = "normal"
    for grp, sub in q_df.groupby("dataset"):
        if len(sub) < 5:
            continue
        p05_sharp = sub["sharpness"].quantile(0.05)
        p05_bright = sub["brightness"].quantile(0.05)
        p95_bright = sub["brightness"].quantile(0.95)

        low_sharp_mask = (q_df["dataset"] == grp) & (q_df["sharpness"] <= p05_sharp)
        dark_mask = (q_df["dataset"] == grp) & (q_df["brightness"] <= p05_bright)
        bright_mask = (q_df["dataset"] == grp) & (q_df["brightness"] >= p95_bright)

        q_df.loc[dark_mask, "quality_flag"] = "low_brightness_p05"
        q_df.loc[bright_mask, "quality_flag"] = "high_brightness_p95"
        q_df.loc[low_sharp_mask, "quality_flag"] = "low_sharpness_p05"

    # Build concise summary table per dataset
    summary_rows: List[Dict[str, Any]] = []
    for grp, sub in q_df.groupby("dataset", dropna=False):
        summary_rows.append(
            {
                "dataset": str(grp),
                "analyzed_samples": len(sub),
                "sharpness_mean": round(float(sub["sharpness"].mean()), 2),
                "sharpness_median": round(float(sub["sharpness"].median()), 2),
                "sharpness_p05": round(float(sub["sharpness"].quantile(0.05)), 2),
                "brightness_mean": round(float(sub["brightness"].mean()), 2),
                "brightness_std": round(float(sub["brightness"].std()), 2),
                "contrast_mean": round(float(sub["contrast"].mean()), 2),
                "red_mean": round(float(sub["red_mean"].mean()), 2),
                "green_mean": round(float(sub["green_mean"].mean()), 2),
                "blue_mean": round(float(sub["blue_mean"].mean()), 2),
            }
        )

    return {
        "per_image": q_df,
        "summary": pd.DataFrame(summary_rows),
    }


# ============================================================================
# E. COMPARATIVE & DOMAIN BIAS ANALYSIS
# ============================================================================

def compare_datasets(
    df: pd.DataFrame,
    dim_df: Optional[pd.DataFrame] = None,
    quality_df: Optional[pd.DataFrame] = None,
    group_col: str = "dataset",
) -> pd.DataFrame:
    """Synthesize a unified cross-dataset comparison table (Sizes, Demographics, Optical Quality, Color)."""
    if df.empty:
        return pd.DataFrame()

    g_col = group_col if group_col in df.columns else "data_source"
    work_df = df[df["file_exists"]].copy() if "file_exists" in df.columns else df.copy()
    if work_df.empty:
        work_df = df.copy()

    rows: List[Dict[str, Any]] = []
    for grp, sub in work_df.groupby(g_col, dropna=False):
        n = len(sub)
        female_pct = round(float((sub["gender_norm"] == "female").mean() * 100.0), 2) if "gender_norm" in sub else np.nan
        male_pct = round(float((sub["gender_norm"] == "male").mean() * 100.0), 2) if "gender_norm" in sub else np.nan
        young_pct = round(float((sub["young_norm"] == "young").mean() * 100.0), 2) if "young_norm" in sub else np.nan
        glasses_pct = (
            round(float((sub["glasses_norm"] == "with_glasses").mean() * 100.0), 2)
            if "glasses_norm" in sub
            else np.nan
        )

        rec: Dict[str, Any] = {
            "group": str(grp),
            "available_images": n,
            "female_pct": female_pct,
            "male_pct": male_pct,
            "young_pct": young_pct,
            "with_glasses_pct": glasses_pct,
        }

        if dim_df is not None and not dim_df.empty and g_col in dim_df.columns:
            d_sub = dim_df[dim_df[g_col] == grp]
            if not d_sub.empty:
                rec["dominant_resolution"] = d_sub["resolution_str"].mode().iloc[0]
                rec["mean_aspect_ratio"] = round(float(d_sub["aspect_ratio"].mean()), 2)

        if quality_df is not None and not quality_df.empty and "dataset" in quality_df.columns:
            q_sub = quality_df[quality_df["dataset"] == grp]
            if not q_sub.empty:
                rec["mean_sharpness"] = round(float(q_sub["sharpness"].mean()), 2)
                rec["mean_brightness"] = round(float(q_sub["brightness"].mean()), 2)
                rec["mean_contrast"] = round(float(q_sub["contrast"].mean()), 2)
                rec["rgb_mean_tuple"] = (
                    f"({q_sub['red_mean'].mean():.1f}, "
                    f"{q_sub['green_mean'].mean():.1f}, "
                    f"{q_sub['blue_mean'].mean():.1f})"
                )

        rows.append(rec)

    return pd.DataFrame(rows).reset_index(drop=True)


# ============================================================================
# F. VISUALIZATION FUNCTIONS (RETURN MATPLOTLIB FIGURES)
# ============================================================================

def plot_dataset_distribution(
    summary_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot Bar Chart (Manifest vs. Available on Disk) and Donut Chart (Available Share)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if summary_df.empty:
        axes[0].text(0.5, 0.5, "No data available", ha="center", va="center")
        axes[1].axis("off")
        return fig

    groups = summary_df["group"].astype(str).tolist()
    x = np.arange(len(groups))
    width = 0.36

    b1 = axes[0].bar(x - width / 2, summary_df["manifest_records"], width, label="Manifest Target", color="#95a5a6")
    b2 = axes[0].bar(x + width / 2, summary_df["existing_files"], width, label="Available on Disk", color="#2980b9")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, fontweight="bold")
    axes[0].set_ylabel("Number of Images (count)")
    axes[0].set_title("Dataset Inventory: Manifest Target vs. Available on Disk", fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(
                f"{int(h)}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    avail_counts = summary_df["existing_files"].to_numpy()
    if avail_counts.sum() > 0:
        colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12"][: len(groups)]
        axes[1].pie(
            avail_counts,
            labels=groups,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"width": 0.42, "edgecolor": "white"},
        )
        axes[1].set_title("Available Dataset Share on Disk (%)", fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "0 existing files on disk", ha="center", va="center")
        axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_demographics(
    demo_results: Dict[str, pd.DataFrame],
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot 2x2 Demographic Panel: Gender, Glasses, FFHQ Granular Age, and Head Pose Angles."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Harmonized Gender
    gender_df = demo_results.get("gender", pd.DataFrame())
    if not gender_df.empty:
        piv_g = gender_df.pivot(index="dataset", columns="gender_norm", values="percentage").fillna(0.0)
        piv_g.plot(kind="bar", ax=axes[0, 0], rot=0, colormap="Paired", edgecolor="black")
        axes[0, 0].set_title("Harmonized Gender Distribution by Dataset (%)", fontweight="bold")
        axes[0, 0].set_ylabel("Percentage (%)")
        axes[0, 0].set_xlabel("Dataset Source")
        axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
        axes[0, 0].legend(title="Gender")
    else:
        axes[0, 0].text(0.5, 0.5, "No gender metadata", ha="center", va="center")

    # 2. Harmonized Glasses / Occlusion
    gl_df = demo_results.get("glasses_harmonized", pd.DataFrame())
    if not gl_df.empty:
        piv_gl = gl_df.pivot(index="dataset", columns="glasses_norm", values="percentage").fillna(0.0)
        piv_gl.plot(kind="bar", stacked=True, ax=axes[0, 1], rot=0, color=["#e67e22", "#bdc3c7", "#3498db"], edgecolor="black")
        axes[0, 1].set_title("Eyeglasses / Ocular Occlusion Proportion (%)", fontweight="bold")
        axes[0, 1].set_ylabel("Percentage (%)")
        axes[0, 1].set_xlabel("Dataset Source")
        axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
        axes[0, 1].legend(title="Glasses Status")
    else:
        axes[0, 1].text(0.5, 0.5, "No glasses metadata", ha="center", va="center")

    # 3. Granular Age Group (FFHQ) vs Harmonized Young Status
    age_gran = demo_results.get("age_granular", pd.DataFrame())
    ffhq_age = age_gran[
        (age_gran["dataset"] == "ffhq") & (~age_gran["age_group_raw"].isin(["not_applicable", "missing"]))
    ] if not age_gran.empty else pd.DataFrame()

    if not ffhq_age.empty:
        age_order = ["0-2", "3-6", "7-9", "10-14", "15-19", "20-29", "30-39", "40-49", "50-69", "70+"]
        ffhq_age_sorted = ffhq_age.set_index("age_group_raw").reindex(
            [a for a in age_order if a in ffhq_age["age_group_raw"].values]
        ).reset_index()
        axes[1, 0].bar(ffhq_age_sorted["age_group_raw"], ffhq_age_sorted["count"], color="#27ae60", edgecolor="black")
        axes[1, 0].set_title("FFHQ Granular Age Group Distribution (Count)", fontweight="bold")
        axes[1, 0].set_xlabel("Age Group (Years)")
        axes[1, 0].set_ylabel("Image Count")
        axes[1, 0].tick_params(axis="x", rotation=30)
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    else:
        age_harm = demo_results.get("age_harmonized", pd.DataFrame())
        if not age_harm.empty:
            piv_a = age_harm.pivot(index="dataset", columns="young_norm", values="percentage").fillna(0.0)
            piv_a.plot(kind="bar", ax=axes[1, 0], rot=0)
            axes[1, 0].set_title("Harmonized Age Cohort (Young vs. Non-Young %)", fontweight="bold")
            axes[1, 0].set_ylabel("Percentage (%)")

    # 4. Harmonized Young vs Non-Young Cross-Dataset Comparison
    age_harm = demo_results.get("age_harmonized", pd.DataFrame())
    if not age_harm.empty:
        piv_y = age_harm.pivot(index="dataset", columns="young_norm", values="percentage").fillna(0.0)
        piv_y.plot(kind="bar", ax=axes[1, 1], rot=0, color=["#9b59b6", "#1abc9c", "#95a5a6"], edgecolor="black")
        axes[1, 1].set_title("Cross-Dataset Age Cohort Comparison (Young vs. Non-Young %)", fontweight="bold")
        axes[1, 1].set_ylabel("Percentage (%)")
        axes[1, 1].set_xlabel("Dataset Source")
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
        axes[1, 1].legend(title="Cohort")
    else:
        axes[1, 1].text(0.5, 0.5, "No age cohort metadata", ha="center", va="center")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_image_properties(
    dim_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot Width/Height scatter, Resolution group bar chart, and File Size distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    if dim_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No dimension data", ha="center", va="center")
        return fig

    palette = {"ffhq": "#2980b9", "celeba_hq": "#e74c3c"}
    for grp, sub in dim_df.groupby("dataset", dropna=False):
        c = palette.get(str(grp), "#2c3e50")
        axes[0].scatter(sub["width"], sub["height"], alpha=0.6, label=str(grp), color=c, s=30)
    axes[0].set_title("Image Width vs. Height (Pixels)", fontweight="bold")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.4)

    res_counts = dim_df.groupby(["dataset", "resolution_str"]).size().unstack(fill_value=0)
    res_counts.plot(kind="bar", ax=axes[1], rot=0, edgecolor="black")
    axes[1].set_title("Resolution Group Counts by Dataset", fontweight="bold")
    axes[1].set_xlabel("Dataset Source")
    axes[1].set_ylabel("Image Count")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for grp, sub in dim_df.groupby("dataset", dropna=False):
        c = palette.get(str(grp), "#8e44ad")
        axes[2].hist(sub["file_size_kb"], bins=25, alpha=0.6, label=str(grp), color=c, edgecolor="black")
    axes[2].set_title("File Size Distribution (KB)", fontweight="bold")
    axes[2].set_xlabel("File Size (KB)")
    axes[2].set_ylabel("Frequency (Images)")
    axes[2].legend()
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_image_quality(
    quality_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot 2x2 Optical & Color Quality Panel: Sharpness, Brightness, Contrast, and RGB Channels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    if quality_df.empty:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, "No quality data", ha="center", va="center")
        return fig

    palette = {"ffhq": "#2980b9", "celeba_hq": "#e74c3c"}

    # 1. Sharpness Histogram
    for grp, sub in quality_df.groupby("dataset", dropna=False):
        axes[0, 0].hist(
            sub["sharpness"],
            bins=30,
            alpha=0.6,
            label=f"{grp} (median={sub['sharpness'].median():.1f})",
            color=palette.get(str(grp), "#34495e"),
            edgecolor="black",
        )
    axes[0, 0].set_title("Sharpness Distribution (Laplacian Variance)", fontweight="bold")
    axes[0, 0].set_xlabel("Laplacian Variance (Var(∇²I))")
    axes[0, 0].set_ylabel("Image Count")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # 2. Brightness Histogram
    for grp, sub in quality_df.groupby("dataset", dropna=False):
        axes[0, 1].hist(
            sub["brightness"],
            bins=30,
            alpha=0.6,
            label=f"{grp} (mean={sub['brightness'].mean():.1f})",
            color=palette.get(str(grp), "#16a085"),
            edgecolor="black",
        )
    axes[0, 1].set_title("Brightness Distribution (Mean Luminance [0-255])", fontweight="bold")
    axes[0, 1].set_xlabel("Mean Pixel Intensity")
    axes[0, 1].set_ylabel("Image Count")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Contrast Boxplot
    groups = [str(g) for g, _ in quality_df.groupby("dataset", dropna=False)]
    contrast_data = [sub["contrast"].dropna().values for _, sub in quality_df.groupby("dataset", dropna=False)]
    bp = axes[1, 0].boxplot(contrast_data, patch_artist=True)
    axes[1, 0].set_xticks(range(1, len(groups) + 1))
    axes[1, 0].set_xticklabels(groups)
    for patch, grp in zip(bp["boxes"], groups):
        patch.set_facecolor(palette.get(grp, "#9b59b6"))
        patch.set_alpha(0.65)
    axes[1, 0].set_title("RMS Contrast Comparison Across Datasets", fontweight="bold")
    axes[1, 0].set_xlabel("Dataset Source")
    axes[1, 0].set_ylabel("Pixel Intensity Std Dev (σ_Y)")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)

    # 4. RGB Mean Channel Intensity Comparison
    rgb_means = quality_df.groupby("dataset")[["red_mean", "green_mean", "blue_mean"]].mean()
    rgb_means.plot(
        kind="bar",
        ax=axes[1, 1],
        rot=0,
        color=["#e74c3c", "#2ecc71", "#3498db"],
        edgecolor="black",
    )
    axes[1, 1].set_title("Mean RGB Channel Intensity Comparison (0-255)", fontweight="bold")
    axes[1, 1].set_xlabel("Dataset Source")
    axes[1, 1].set_ylabel("Mean Channel Intensity")
    axes[1, 1].legend(["Red Channel", "Green Channel", "Blue Channel"])
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_dataset_comparison(
    quality_df: pd.DataFrame,
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot Side-by-Side Normalized Boxplots & Feature Correlation Heatmap for Domain Bias Analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    if quality_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No comparison data", ha="center", va="center")
        return fig

    metrics = ["sharpness", "brightness", "contrast", "red_mean", "green_mean", "blue_mean"]
    avail_metrics = [m for m in metrics if m in quality_df.columns]

    # Left: Z-score normalized means per dataset to compare multi-scale features on one chart
    norm_df = quality_df.copy()
    for m in avail_metrics:
        std = norm_df[m].std()
        norm_df[m] = (norm_df[m] - norm_df[m].mean()) / (std if std > 0 else 1.0)

    grp_means = norm_df.groupby("dataset")[avail_metrics].mean().T
    grp_means.plot(kind="bar", ax=axes[0], rot=25, edgecolor="black")
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_title("Standardized Feature Signatures (Z-Score Means by Dataset)", fontweight="bold")
    axes[0].set_ylabel("Mean Z-Score (Standardized Units)")
    axes[0].set_xlabel("Optical & Color Features")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Right: Correlation matrix across optical & color metrics
    corr = quality_df[avail_metrics].corr()
    im = axes[1].imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[1].set_xticks(range(len(avail_metrics)))
    axes[1].set_yticks(range(len(avail_metrics)))
    axes[1].set_xticklabels(avail_metrics, rotation=30, ha="right")
    axes[1].set_yticklabels(avail_metrics)
    axes[1].set_title("Optical & RGB Metric Correlation Matrix", fontweight="bold")

    for i in range(len(avail_metrics)):
        for j in range(len(avail_metrics)):
            axes[1].text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr.values[i, j]) > 0.5 else "black",
                fontsize=8.5,
            )
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def display_image_gallery(
    df: pd.DataFrame,
    n_per_group: int = 4,
    group_col: str = "dataset",
    random_state: int = 42,
    title: str = "Representative Visual Image Gallery",
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Render a clean, reproducible visual gallery grid with official metadata annotations."""
    existing_df = df[df["file_exists"]].copy() if "file_exists" in df.columns else df.copy()
    if existing_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No existing images available for gallery", ha="center", va="center")
        ax.axis("off")
        return fig

    g_col = group_col if group_col in existing_df.columns else "data_source"
    groups = [g for g, _ in existing_df.groupby(g_col, dropna=False)]
    n_rows = max(1, len(groups))
    n_cols = max(1, n_per_group)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes_grid = np.array([[axes]])
    elif n_rows == 1:
        axes_grid = np.array([axes])
    elif n_cols == 1:
        axes_grid = np.array([[ax] for ax in axes])
    else:
        axes_grid = axes

    for r_idx, grp in enumerate(groups):
        sub = existing_df[existing_df[g_col] == grp]
        sampled = sub.sample(n=min(len(sub), n_cols), random_state=random_state) if not sub.empty else sub

        for c_idx in range(n_cols):
            ax = axes_grid[r_idx, c_idx]
            if c_idx < len(sampled):
                row = sampled.iloc[c_idx]
                try:
                    with Image.open(row["resolved_path"]) as im:
                        ax.imshow(im.convert("RGB"))
                    g_label = row.get("gender_norm", "N/A")
                    a_label = (
                        row.get("age_group_raw")
                        if row.get("age_group_raw") not in {"not_applicable", "missing", "unknown"}
                        else row.get("young_norm", "N/A")
                    )
                    gl_label = row.get("glasses_detail", "N/A")
                    ax.set_title(
                        f"{row['image_id']}\n[{grp}] {g_label} | {a_label}\nGlasses: {gl_label}",
                        fontsize=8.5,
                    )
                except Exception as exc:
                    ax.text(0.5, 0.5, f"Read Error:\n{exc}", ha="center", va="center", fontsize=8)
            ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ============================================================================
# G. REPORTING & EXPORT
# ============================================================================

def generate_eda_summary(
    summary_df: pd.DataFrame,
    demo_results: Dict[str, pd.DataFrame],
    res_df: pd.DataFrame,
    quality_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    duplicates_dict: Optional[Dict[str, pd.DataFrame]] = None,
) -> str:
    """Generate an evidence-based Markdown EDA report strictly from computed tables."""
    lines: List[str] = [
        "# UTH Computer Vision Coursework — Exploratory Data Analysis (EDA) Report",
        "",
        "## 1. Executive Summary & Dataset Inventory",
    ]

    if summary_df.empty:
        lines.append("- No valid records were found in the provided manifest.")
    else:
        total_manifest = int(summary_df["manifest_records"].sum())
        total_avail = int(summary_df["existing_files"].sum())
        total_missing = int(summary_df["missing_files"].sum())
        lines.extend(
            [
                f"- **Total Manifest Records Indexed:** {total_manifest}",
                f"- **Available Images Verified on Disk:** {total_avail} ({total_avail / max(1, total_manifest) * 100:.2f}%)",
                f"- **Missing Files on Disk:** {total_missing}",
                "",
                "```",
                summary_df.to_string(index=False),
                "```",
            ]
        )

    lines.extend(["", "## 2. Image Resolution & Geometric Integrity"])
    if not res_df.empty:
        lines.extend(["```", res_df.to_string(index=False), "```"])
    else:
        lines.append("- Resolution analysis was skipped or no existing images were available.")

    lines.extend(["", "## 3. Optical Image Quality & Duplicate Audit"])
    if not quality_summary_df.empty:
        lines.extend(["```", quality_summary_df.to_string(index=False), "```"])
    else:
        lines.append("- Optical quality analysis was not run.")

    if duplicates_dict:
        n_exact = len(duplicates_dict.get("exact_duplicates", pd.DataFrame()))
        n_perc = len(duplicates_dict.get("perceptual_duplicates", pd.DataFrame()))
        lines.append(f"- **Exact Byte Duplicates (MD5):** {n_exact} records flagged.")
        lines.append(f"- **Near-Perceptual Duplicates (64-bit dHash):** {n_perc} records flagged.")

    lines.extend(["", "## 4. Cross-Dataset Comparison & Domain Bias Assessment"])
    if not comparison_df.empty:
        lines.extend(["```", comparison_df.to_string(index=False), "```"])


    return "\n".join(lines) + "\n"


def export_eda_results(
    output_dir: PathLike,
    tables: Dict[str, pd.DataFrame],
    figures: Dict[str, plt.Figure],
    report_markdown: str,
    quality_audit_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Path]:
    """Export all tables, figures, quality audit CSVs, and Markdown report to structured directories.

    Target structure:
        <output_dir>/
        ├── figures/
        ├── tables/
        ├── reports/
        └── quality_audit/
    """
    base_dir = Path(output_dir)
    fig_dir = base_dir / "figures"
    tbl_dir = base_dir / "tables"
    rep_dir = base_dir / "reports"
    aud_dir = base_dir / "quality_audit"

    for d in (fig_dir, tbl_dir, rep_dir, aud_dir):
        d.mkdir(parents=True, exist_ok=True)

    exported_paths: Dict[str, Path] = {}

    for name, df in tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            p = tbl_dir / f"{name}.csv"
            df.to_csv(p, index=False, encoding="utf-8")
            exported_paths[f"table:{name}"] = p

    if quality_audit_tables:
        for name, df in quality_audit_tables.items():
            if isinstance(df, pd.DataFrame):
                p = aud_dir / f"{name}.csv"
                df.to_csv(p, index=False, encoding="utf-8")
                exported_paths[f"audit:{name}"] = p

    for name, fig in figures.items():
        if fig is not None:
            p = fig_dir / f"{name}.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            exported_paths[f"figure:{name}"] = p

    report_path = rep_dir / "EDA_SUMMARY_REPORT.md"
    report_path.write_text(report_markdown, encoding="utf-8")
    exported_paths["report:markdown"] = report_path

    logger.info("Exported %d EDA artifacts to %s", len(exported_paths), base_dir)
    return exported_paths