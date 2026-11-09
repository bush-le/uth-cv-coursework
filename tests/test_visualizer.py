"""Unit tests for visualization utilities."""

from pathlib import Path
import numpy as np
from PIL import Image

from src.common.visualizer import ensure_rgb, plot_real_vs_synthetic_comparison, save_image_grid


def test_ensure_rgb():
    """Verify BGR to RGB color inversion."""
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # Blue in BGR
    rgb = ensure_rgb(bgr, source_format="bgr")
    assert rgb[0, 0, 2] == 255  # Blue in RGB is channel 2
    assert rgb[0, 0, 0] == 0


def test_save_image_grid(tmp_path: Path):
    """Verify grid saving without errors."""
    out_file = tmp_path / "grid.png"
    imgs = [np.ones((32, 32, 3), dtype=np.uint8) * 128 for _ in range(4)]
    save_image_grid(imgs, out_file, cols=2, titles=["1", "2", "3", "4"])
    assert out_file.exists()


def test_plot_real_vs_synthetic_comparison(tmp_path: Path):
    """Verify 3-row comparison gallery output creation."""
    real_f = tmp_path / "real.png"
    fake_f = tmp_path / "fake.png"
    hard_f = tmp_path / "hard.png"

    img = Image.new("RGB", (64, 64), color="blue")
    img.save(real_f)
    img.save(fake_f)
    img.save(hard_f)

    gallery_out = tmp_path / "gallery.png"
    plot_real_vs_synthetic_comparison(
        real_paths=[real_f],
        synthetic_paths=[fake_f],
        hard_paths=[hard_f],
        output_path=gallery_out,
        num_samples=1,
    )
    assert gallery_out.exists()
