"""Tests for coursework notebook migration and legacy archiving structure."""

from pathlib import Path
import pytest


@pytest.mark.smoke
def test_legacy_v1_directory_structure() -> None:
    legacy_dir = Path("notebooks/coursework/legacy_v1")
    assert legacy_dir.is_dir(), "legacy_v1 directory must exist"
    readme_path = legacy_dir / "README.md"
    assert readme_path.is_file(), "legacy_v1/README.md must exist"

    expected_files = [
        "00_master_deepfake_defense_coursework.ipynb",
        "01_eda_faces_demographics.ipynb",
        "02_sd_generation_and_curation.ipynb",
        "03_hard_samples_and_benchmark_comparison.ipynb",
        "04_active_learning_production_visualizer.ipynb",
        "05_synthetic_face_6k_realimage_benchmark.ipynb",
        "06_specialization_vs_generalization_synthesis.ipynb",
        "07_universal_deepfake_benchmark.ipynb",
        "08_champion_m3_dual_engine_quad_vault_benchmark.ipynb",
        "generator_realistic_vision_kaggle.ipynb",
    ]
    for fname in expected_files:
        assert (legacy_dir / fname).is_file(), f"Missing archived file: {fname}"
