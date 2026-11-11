"""Unit tests for visualization utilities."""

from pathlib import Path
import numpy as np
from PIL import Image

from src.common.visualizer import (
    ensure_rgb,
    plot_adversarial_trajectory_comparison,
    plot_clean_10k_dataset_dashboard,
    plot_comprehensive_metrics_grid,
    plot_error_attribution_and_mode_quotas,
    plot_legacy_6k_benchmark,
    plot_legacy_6k_distributions,
    plot_legacy_6k_gallery_inspection,
    plot_legacy_sample_inspection,
    plot_real_vs_synthetic_comparison,
    plot_tri_vault_tradeoff_analysis,
    render_golden_comparison_markdown,
    render_legacy_6k_markdown_table,
    render_tri_vault_summary_table,
    plot_loss_and_roc_curves_comparison,
    save_image_grid,
)


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


def test_production_visualizer_plots(tmp_path: Path):
    """Verify production visualizer dashboard functions on actual experiment artifacts."""
    report_file = Path("experiments/logs/clean_10k_production_report.json")
    benchmark_file = Path("experiments/logs/model_comparison_benchmark.json")
    metrics_file = Path("experiments/logs/closed_loop_metrics.json")

    if report_file.exists():
        fig1 = plot_clean_10k_dataset_dashboard(report_file, output_path=tmp_path / "dash.png")
        assert (tmp_path / "dash.png").exists()

    if benchmark_file.exists():
        fig2 = plot_adversarial_trajectory_comparison(benchmark_file, output_path=tmp_path / "traj.png")
        assert (tmp_path / "traj.png").exists()

    if metrics_file.exists():
        fig3 = plot_error_attribution_and_mode_quotas(metrics_file, output_path=tmp_path / "quotas.png")
        assert (tmp_path / "quotas.png").exists()

    if benchmark_file.exists():
        fig4 = plot_comprehensive_metrics_grid(benchmark_file, output_path=tmp_path / "comp_grid.png")
        assert (tmp_path / "comp_grid.png").exists()

        md_text = render_golden_comparison_markdown(benchmark_file)
        assert "Cumulative Challenge Test Vault" in md_text
        assert "Master Frozen Test Vault" in md_text

    legacy_file = Path("experiments/logs/legacy_6k_benchmark_results.json")
    if legacy_file.exists():
        fig5 = plot_legacy_6k_benchmark(legacy_file, output_path=tmp_path / "legacy_6k.png")
        assert (tmp_path / "legacy_6k.png").exists()

        md_table = render_legacy_6k_markdown_table(legacy_file)
        assert "Báo Cáo Kiểm Định Độc Lập Toàn Bộ 6.000 Ảnh Ngoại Miền" in md_table
        assert "$M_0$ (Baseline D0)" in md_table

    npz_file = Path("experiments/logs/legacy_6k_predictions.npz")
    if npz_file.exists():
        fig6 = plot_legacy_6k_distributions(npz_file, output_path=tmp_path / "legacy_dist.png")
        assert (tmp_path / "legacy_dist.png").exists()

        fig7 = plot_legacy_6k_gallery_inspection(npz_file, output_path=tmp_path / "legacy_gal.png", num_samples=3)
        assert (tmp_path / "legacy_gal.png").exists()

        fig8 = plot_legacy_sample_inspection(npz_file, sample_idx=0, output_path=tmp_path / "legacy_single.png")
        assert (tmp_path / "legacy_single.png").exists()

    if benchmark_file.exists():
        fig9 = plot_tri_vault_tradeoff_analysis(
            challenge_benchmark_path=benchmark_file,
            legacy_6k_benchmark_path=legacy_file,
            npz_path=npz_file,
            output_path=tmp_path / "tri_vault.png",
        )
        assert (tmp_path / "tri_vault.png").exists()

        md_tri = render_tri_vault_summary_table(
            challenge_benchmark_path=benchmark_file,
            legacy_6k_benchmark_path=legacy_file,
            npz_path=npz_file,
        )
        assert "Multi-Vault Scientific Ledger" in md_tri

        fig_loss_roc = plot_loss_and_roc_curves_comparison(
            comparison_path=benchmark_file,
            output_path=tmp_path / "loss_roc.png",
        )
        assert (tmp_path / "loss_roc.png").exists()






