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
