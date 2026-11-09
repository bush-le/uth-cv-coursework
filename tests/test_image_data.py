"""Unit tests for 2D deepfake image data contracts, schema and datamodule."""

import pytest
import torch

from src.data.image_datamodule import DeepfakeImageDataModule
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_schema import ImageMetadata, ImageSample


def test_image_metadata_serialization() -> None:
    """Test ImageMetadata serialization to/from dictionary including hard-sample fields."""
    meta = ImageMetadata(
        image_id="face_001",
        label=1,
        manipulation_type="sd_text2img",
        prompt="a portrait of a person",
        source_real_id="real_001",
        file_path="data/processed/images/face_001.png",
        is_hard_sample=True,
        baseline_score=0.22,
        data_source="hard_sample",
        demographics={"age_group": "50-69", "gender": "male", "glasses": "Dark"},
    )
    d = meta.to_dict()
    assert d["image_id"] == "face_001"
    assert d["label"] == 1
    assert d["manipulation_type"] == "sd_text2img"
    assert d["is_hard_sample"] is True
    assert d["baseline_score"] == pytest.approx(0.22)
    assert d["data_source"] == "hard_sample"
    assert d["demographics"]["age_group"] == "50-69"

    restored = ImageMetadata.from_dict(d)
    assert restored.image_id == meta.image_id
    assert restored.label == meta.label
    assert restored.is_hard_sample is True
    assert restored.baseline_score == pytest.approx(0.22)
    assert restored.data_source == "hard_sample"
    assert restored.demographics == {"age_group": "50-69", "gender": "male", "glasses": "Dark"}


def test_image_sample_contract() -> None:
    """Test ImageSample tensor shape contracts (C, H, W)."""
    img = torch.rand(3, 256, 256, dtype=torch.float32)
    meta = ImageMetadata(image_id="face_002", label=0, is_hard_sample=True)
    sample = ImageSample(image=img, metadata=meta)

    assert sample.image.shape == (3, 256, 256)
    assert sample.label == 0
    assert sample.is_hard_sample is True

    with pytest.raises(ValueError):
        # 2D instead of 3D (C, H, W)
        ImageSample(image=torch.rand(256, 256), metadata=meta)


def test_deepfake_image_dataset_mock() -> None:
    """Test DeepfakeImageDataset loading with in-memory cache."""
    metadata_list = [
        ImageMetadata(image_id=f"img_{i}", label=i % 2, manipulation_type="sd_inpaint")
        for i in range(10)
    ]
    cache = {f"img_{i}": torch.rand(3, 128, 128) for i in range(10)}

    dataset = DeepfakeImageDataset(metadata_list=metadata_list, cache=cache)
    assert len(dataset) == 10
    sample = dataset[0]
    assert isinstance(sample, ImageSample)
    assert sample.image.shape == (3, 128, 128)
    assert sample.label in (0, 1)


def test_deepfake_image_datamodule_batching_and_identity_disjoint() -> None:
    """Test DeepfakeImageDataModule batch collation, hard-sample oversampling, and identity-disjoint split."""
    metadata_list = [
        ImageMetadata(
            image_id=f"img_{i}",
            label=i % 2,
            source_real_id=f"id_{i // 2}",
            is_hard_sample=(i == 1),
            data_source="hard_sample" if i == 1 else "baseline",
        )
        for i in range(16)
    ]
    cache = {f"img_{i}": torch.rand(3, 64, 64) for i in range(16)}

    dm = DeepfakeImageDataModule(
        metadata_list=metadata_list,
        cache=cache,
        batch_size=4,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
        hard_sample_oversample_factor=2,
    )
    dm.setup()

    train_ids = {m.source_real_id for m in dm.train_dataset.metadata_list}
    val_ids = {m.source_real_id for m in dm.val_dataset.metadata_list}
    test_ids = {m.source_real_id for m in dm.test_dataset.metadata_list}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert "images" in batch
    assert "labels" in batch
    assert "is_hard_sample" in batch
    assert batch["images"].shape == (4, 3, 64, 64)
    assert batch["labels"].shape == (4,)
    assert batch["is_hard_sample"].shape == (4,)
