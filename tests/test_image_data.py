"""Unit tests for 2D deepfake image data contracts, schema and datamodule."""

import pytest
import torch
from src.data.image_schema import ImageMetadata, ImageSample
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_datamodule import DeepfakeImageDataModule


def test_image_metadata_serialization():
    """Test ImageMetadata serialization to/from dictionary."""
    meta = ImageMetadata(
        image_id="face_001",
        label=1,
        manipulation_type="sd_text2img",
        prompt="a portrait of a person",
        source_real_id="real_001",
        file_path="data/processed/images/face_001.png",
    )
    d = meta.to_dict()
    assert d["image_id"] == "face_001"
    assert d["label"] == 1
    assert d["manipulation_type"] == "sd_text2img"

    restored = ImageMetadata.from_dict(d)
    assert restored.image_id == meta.image_id
    assert restored.label == meta.label


def test_image_sample_contract():
    """Test ImageSample tensor shape contracts (C, H, W)."""
    img = torch.rand(3, 256, 256, dtype=torch.float32)
    meta = ImageMetadata(image_id="face_002", label=0)
    sample = ImageSample(image=img, metadata=meta)

    assert sample.image.shape == (3, 256, 256)
    assert sample.label == 0

    with pytest.raises(ValueError):
        # 2D instead of 3D (C, H, W)
        ImageSample(image=torch.rand(256, 256), metadata=meta)


def test_deepfake_image_dataset_mock():
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


def test_deepfake_image_datamodule_batching():
    """Test DeepfakeImageDataModule batch collation yielding (B, 3, H, W) and (B,)."""
    metadata_list = [
        ImageMetadata(image_id=f"img_{i}", label=i % 2)
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
    )
    dm.setup()
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    assert "images" in batch
    assert "labels" in batch
    assert batch["images"].shape == (4, 3, 64, 64)
    assert batch["labels"].shape == (4,)
