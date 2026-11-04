"""Unit tests for paired facial occlusion dataset and transforms."""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.occlusion_dataset import FacialOcclusionDataset


def test_facial_occlusion_dataset_dummy(tmp_path):
    img_path = tmp_path / "face.png"
    mask_path = tmp_path / "mask.png"

    # Create dummy 512x512 RGB image and uint8 mask
    Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(img_path)
    # Mask with class 1 (mask) in center
    mask_arr = np.zeros((512, 512), dtype=np.uint8)
    mask_arr[100:300, 100:300] = 1
    Image.fromarray(mask_arr).save(mask_path)

    dataset = FacialOcclusionDataset(
        image_paths=[img_path],
        mask_paths=[mask_path],
        image_size=(512, 512),
        is_train=False,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    assert "image" in sample
    assert "mask" in sample

    assert isinstance(sample["image"], torch.Tensor)
    assert isinstance(sample["mask"], torch.Tensor)
    assert sample["image"].shape == (3, 512, 512)
    assert sample["mask"].shape == (512, 512)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.int64
    assert torch.all(sample["mask"][100:300, 100:300] == 1)
