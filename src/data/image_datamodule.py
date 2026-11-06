"""DataModule managing deepfake image dataset splits and batch loading."""

from typing import Any, Callable, Dict, List, Optional
import torch
from torch.utils.data import DataLoader, random_split

from src.common.logger import get_logger
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_schema import ImageMetadata, ImageSample

logger = get_logger(__name__)


def collate_image_samples(samples: List[ImageSample]) -> Dict[str, Any]:
    """Collate list of ImageSample into a batched dictionary.
    
    Returns:
        dict containing:
            'images': torch.Tensor of shape (B, 3, H, W)
            'labels': torch.Tensor of shape (B,)
            'image_ids': List[str]
            'metadata': List[ImageMetadata]
    """
    images = torch.stack([s.image for s in samples], dim=0) # (B, 3, H, W)
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    image_ids = [s.metadata.image_id for s in samples]
    metadatas = [s.metadata for s in samples]

    return {
        "images": images,
        "labels": labels,
        "image_ids": image_ids,
        "metadata": metadatas,
    }


class DeepfakeImageDataModule:
    """DataModule managing deepfake image dataset train/val/test splits."""

    def __init__(
        self,
        metadata_list: List[ImageMetadata],
        cache: Optional[Dict[str, torch.Tensor]] = None,
        train_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        val_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.metadata_list = metadata_list
        self.cache = cache
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.train_dataset: Optional[DeepfakeImageDataset] = None
        self.val_dataset: Optional[DeepfakeImageDataset] = None
        self.test_dataset: Optional[DeepfakeImageDataset] = None

    def setup(self) -> None:
        """Partition metadata into train, val, and test splits deterministically."""
        total = len(self.metadata_list)
        train_len = int(total * self.train_ratio)
        val_len = int(total * self.val_ratio)
        test_len = total - train_len - val_len

        generator = torch.Generator().manual_seed(self.seed)
        train_indices, val_indices, test_indices = random_split(
            range(total), [train_len, val_len, test_len], generator=generator
        )

        train_meta = [self.metadata_list[i] for i in train_indices]
        val_meta = [self.metadata_list[i] for i in val_indices]
        test_meta = [self.metadata_list[i] for i in test_indices]

        self.train_dataset = DeepfakeImageDataset(
            metadata_list=train_meta,
            cache=self.cache,
            transform=self.train_transform,
        )
        self.val_dataset = DeepfakeImageDataset(
            metadata_list=val_meta,
            cache=self.cache,
            transform=self.val_transform,
        )
        self.test_dataset = DeepfakeImageDataset(
            metadata_list=test_meta,
            cache=self.cache,
            transform=self.val_transform,
        )
        logger.info(
            f"ImageDataModule setup complete: {train_len} train, {val_len} val, {test_len} test samples."
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting DataLoader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_image_samples,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting DataLoader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_image_samples,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup() must be called before requesting DataLoader.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_image_samples,
        )
