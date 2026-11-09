"""Centralized configuration loader, validator, and CLI resolver for Dataset Factory."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import yaml

from src.common.logger import get_logger

logger = get_logger(__name__)

REQUIRED_KEYS: Tuple[str, ...] = (
    "paths.raw_manifest",
    "paths.synthetic_dir",
    "paths.fake_manifest",
    "paths.curated_manifest",
    "paths.curation_report",
    "paths.hard_dir",
    "paths.hard_manifest",
    "paths.annotated_manifest",
    "paths.benchmarks_dir",
    "paths.stats_file",
    "paths.gallery_image",
    "seed",
    "generation.model_id",
    "generation.inpaint_model_id",
    "generation.modes",
    "generation.num_per_real",
    "generation.max_real_samples",
    "generation.steps",
    "generation.guidance_scale",
    "generation.img2img_strength",
    "generation.image_size",
    "generation.negative_prompt",
    "generation.inpaint_mask_box",
    "curation.reference_quantile_sharpness",
    "curation.clip.enabled",
    "curation.clip.model_id",
    "curation.clip.min_score",
    "curation.face_check.backend",
    "curation.dedup",
    "hardness.method",
    "hardness.backbone_id",
    "hardness.k_neighbors",
    "hardness.reference_quantile",
    "hardness.detector_checkpoint",
    "hardness.detector_tau",
    "hardness.batch_size",
    "benchmarks.split_ratios",
    "benchmarks.stages",
)


def load_config(path: Union[str, Path] = "configs/dataset_factory.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file into dictionary.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Nested configuration dictionary.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def cfg_get(cfg: Dict[str, Any], dotted_key: str) -> Any:
    """Retrieve value from nested dictionary using dotted key string.

    Args:
        cfg: Configuration dictionary.
        dotted_key: Dot-delimited key string (e.g. 'generation.model_id').

    Returns:
        The target value.

    Raises:
        KeyError: If any key segment is missing in cfg.
    """
    keys = dotted_key.split(".")
    curr: Any = cfg
    for k in keys:
        if not isinstance(curr, dict) or k not in curr:
            raise KeyError(f"Configuration missing required key segment: '{dotted_key}' (at '{k}')")
        curr = curr[k]
    return curr


def resolve(cli_value: Optional[Any], cfg: Dict[str, Any], dotted_key: str) -> Any:
    """Resolve parameter precedence: use CLI override if provided, else fallback to config.

    Args:
        cli_value: Argument value from CLI parser (or None).
        cfg: Configuration dictionary.
        dotted_key: Dotted key string in configuration dictionary.

    Returns:
        Resolved value.
    """
    if cli_value is not None:
        return cli_value
    return cfg_get(cfg, dotted_key)
