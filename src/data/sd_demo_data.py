"""File contracts for the SD image pilot, independent of diffusion libraries."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


def file_sha256(path: Path) -> str:
    """Hash a file without loading all bytes into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a small JSON artifact on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL, rejecting empty manifests and duplicate image IDs."""
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    ids = [str(item["image_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image_id in manifest.")
    return records


def resolve_image_path(value: str, manifest: Path, data_root: Path = Path(".")) -> Path:
    """Resolve explicit root-relative or manifest-relative image paths."""
    path = Path(value)
    candidates = (
        [path] if path.is_absolute() else [data_root / path, manifest.parent / path]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Image missing: {value}; searched {candidates}")


def load_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """EXIF-correct and resize image to CPU RGB float32 (3,H,W) [0,1].

    size is (width,height); the pilot expects already aligned face crops.
    """
    with Image.open(path) as image:
        rgb = (
            ImageOps.exif_transpose(image)
            .convert("RGB")
            .resize(size, Image.Resampling.LANCZOS)
        )
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def load_edit_mask(
    size: tuple[int, int],
    path: Path | None = None,
    box: tuple[float, float, float, float] = (0.30, 0.58, 0.70, 0.80),
) -> torch.Tensor:
    """Return CPU mask float32 (1,H,W) [0,1], white=edit.

    Explicit masks are resized nearest-neighbor. Without a mask, draw an ellipse
    in normalized xyxy box on an aligned crop. This is a pilot ROI, not a
    landmark detector or a claim that every face has been anatomically aligned.
    """
    if path is not None:
        with Image.open(path) as image:
            mask = image.convert("L").resize(size, Image.Resampling.NEAREST)
    else:
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Mask box must be normalized xyxy inside [0,1].")
        width, height = size
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).ellipse(
            (x1 * width, y1 * height, x2 * width, y2 * height), fill=255
        )
    return torch.from_numpy(
        np.asarray(mask, dtype=np.float32).copy() / 255.0
    ).unsqueeze(0)
