"""Unit tests for the 2-round iterative evolutionary framework."""

import pytest
from src.data.image_schema import ImageMetadata
from src.pipelines.run_iterative_framework import assemble_evolutionary_stage


def test_replay_buffer_prevents_catastrophic_forgetting():
    """Verify that assemble_evolutionary_stage monotonically retains all prior baseline and hard samples."""
    d0 = [ImageMetadata(image_id=f"base_{i}", label=1, data_source="baseline") for i in range(10)]
    h1 = [ImageMetadata(image_id=f"h1_{i}", label=1, data_source="hard_sample", is_hard_sample=True) for i in range(5)]
    h2 = [ImageMetadata(image_id=f"h2_{i}", label=1, data_source="hard_sample", is_hard_sample=True) for i in range(5)]

    d1 = assemble_evolutionary_stage(d0, [h1])
    d2 = assemble_evolutionary_stage(d0, [h1, h2])

    assert len(d1) == 15
    assert len(d2) == 20
    # Baseline faces must be 100% preserved
    base_ids_in_d2 = {m.image_id for m in d2 if m.data_source == "baseline"}
    assert len(base_ids_in_d2) == 10


def test_four_way_benchmark_schema():
    """Verify formatting of the 4-way comparative Proof-of-Utility benchmark table."""
    from scripts.evaluate_iterative_benchmarks import build_benchmark_summary_table

    metrics = {
        "M0_baseline": {"accuracy": 0.85, "fake_recall": 0.80, "bypass_rate": 0.20, "auc": 0.875, "fpr": 0.10},
        "M_rand": {"accuracy": 0.875, "fake_recall": 0.85, "bypass_rate": 0.15, "auc": 0.900, "fpr": 0.10},
        "M1_hard": {"accuracy": 0.925, "fake_recall": 0.90, "bypass_rate": 0.10, "auc": 0.940, "fpr": 0.05},
        "M2_adaptive": {"accuracy": 0.975, "fake_recall": 0.95, "bypass_rate": 0.05, "auc": 0.985, "fpr": 0.00},
    }
    markdown_table = build_benchmark_summary_table(metrics)
    assert "| Model |" in markdown_table
    assert "M2_adaptive" in markdown_table
    assert "0.985" in markdown_table

