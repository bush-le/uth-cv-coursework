"""Tests for dataset and datamodule components."""

import pytest

from src.data.datamodule import CVDataModule
from src.data.dataset import StandardCVDataset


@pytest.mark.smoke
def test_standard_cv_dataset_len() -> None:
    """Smoke test: verify dataset length calculation."""
    samples = [("img1.jpg", 0), ("img2.jpg", 1)]
    ds = StandardCVDataset(samples=samples)
    assert len(ds) == 2


def test_cv_datamodule_setup() -> None:
    """Verify DataModule setup stage."""
    dm = CVDataModule(data_dir="data/raw", batch_size=4, num_workers=0)
    dm.setup()
    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
