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
