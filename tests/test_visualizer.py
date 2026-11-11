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


def test_notebook_generator_has_no_processed_paths_or_literals(tmp_path: Path):
    """Verify generated coursework notebooks load configurations dynamically without legacy paths."""
    import json
    from scripts.create_coursework_notebooks import main as create_nb_main

    create_nb_main(output_dir=tmp_path)
    nb_files = list(tmp_path.glob("*.ipynb"))
    assert len(nb_files) == 3

    for nb_file in nb_files:
        content = nb_file.read_text(encoding="utf-8")
        assert "data/processed" not in content, f"{nb_file.name} still contains legacy 'data/processed'!"
        assert "load_config(" in content, f"{nb_file.name} must load settings dynamically via load_config()!"

