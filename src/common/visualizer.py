"""Visualization utilities for images, predictions, and bounding boxes."""

from pathlib import Path

import numpy as np


def ensure_rgb(image: np.ndarray, source_format: str = "bgr") -> np.ndarray:
    """Convert image array to RGB format safely.

    Args:
        image: Image array of shape (H, W, 3) or (H, W).
        source_format: 'bgr' (OpenCV default) or 'rgb'.

    Returns:
        RGB image array of shape (H, W, 3) or (H, W).
    """
    if image.ndim == 2:
        return image

    if source_format.lower() == "bgr":
        return image[:, :, ::-1].copy()

    return image.copy()


def save_image_grid(
    images: list[np.ndarray],
    output_path: str | Path,
    cols: int = 4,
    titles: list[str] | None = None,
) -> None:
    """Save a grid of images for qualitative verification.

    Args:
        images: List of images of shape (H, W, C) or (H, W).
        output_path: Path where the plot will be saved.
        cols: Number of grid columns.
        titles: Optional list of titles corresponding to images.
    """
    import matplotlib.pyplot as plt

    n_images = len(images)
    if n_images == 0:
        return

    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.ravel(axes)

    for i in range(len(axes)):
        if i < n_images:
            img = images[i]
            if img.ndim == 2:
                axes[i].imshow(img, cmap="gray")
            else:
                axes[i].imshow(img)

            if titles and i < len(titles):
                axes[i].set_title(titles[i], fontsize=10)
        axes[i].axis("off")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_real_vs_synthetic_comparison(
    real_paths: list[str | Path],
    synthetic_paths: list[str | Path],
    hard_paths: list[str | Path],
    output_path: str | Path = "experiments/predictions/real_vs_synthetic_gallery.png",
    num_samples: int = 4,
) -> None:
    """Generate a 3-row comparison gallery: Real Faces vs Synthetic Faces vs Mined Hard Samples."""
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3.0, 9.0))
    categories = [
        ("Real Baseline Face (label=0)", real_paths, "#2ecc71"),
        ("Synthetic SD Face (label=1)", synthetic_paths, "#e67e22"),
        ("Mined Hard Sample (Bypass)", hard_paths, "#e74c3c"),
    ]

    for row_idx, (cat_label, paths, color) in enumerate(categories):
        for col_idx in range(num_samples):
            ax = axes[row_idx, col_idx] if num_samples > 1 else axes[row_idx]
            if col_idx < len(paths) and Path(paths[col_idx]).exists():
                with Image.open(paths[col_idx]) as img:
                    ax.imshow(img.convert("RGB"))
            else:
                ax.text(0.5, 0.5, "No sample", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(cat_label, fontsize=10, fontweight="bold", color=color)

    plt.suptitle(
        "UTH Topic 11 — Real vs Synthetic vs Hard-Sample Deepfakes",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
