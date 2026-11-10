"""Unit tests for runtime device detection and diffusion memory optimization."""

import pytest
import torch
from src.common.device import get_optimal_device, get_torch_dtype, optimize_pipeline_memory


def test_device_detection() -> None:
    """Verify optimal device returns valid torch.device and dtype matches compute capability."""
    device = get_optimal_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cuda", "cpu")

    dtype = get_torch_dtype(device)
    assert dtype in (torch.float16, torch.float32)
    if device.type == "cuda":
        assert dtype == torch.float16
    else:
        assert dtype == torch.float32


def test_optimize_pipeline_memory_dummy() -> None:
    """Verify memory optimization method safely checks and calls attention slicing."""
    class DummyPipeline:
        def __init__(self):
            self.sliced = False
            self.vae_slicing = False

        def enable_attention_slicing(self, slice_size="auto"):
            self.sliced = True

        def enable_vae_slicing(self):
            self.vae_slicing = True

    dummy = DummyPipeline()
    optimize_pipeline_memory(dummy)
    assert dummy.sliced is True
    assert dummy.vae_slicing is True


def test_temp_directory_configured_safely() -> None:
    """Verify TMPDIR is pointed to project local cache on NVMe SSD, not the restricted tmpfs."""
    import os
    import tempfile
    from pathlib import Path

    tmpdir = os.environ.get("TMPDIR")
    assert tmpdir is not None, "TMPDIR must be explicitly set"
    assert not tmpdir.startswith("/tmp"), f"TMPDIR ({tmpdir}) must not be in /tmp (restricted tmpfs)"
    assert Path(tmpdir).is_dir(), f"TMPDIR directory {tmpdir} must exist"

