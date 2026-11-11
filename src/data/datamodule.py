"""DataModule managing splits, DataLoaders, and pipeline ingestion."""

from pathlib import Path
from typing import Any
import torch

from src.data.dataset import StandardCVDataset
from src.data.transforms import get_base_transforms


class CVDataModule:
    """Orchestrates datasets and DataLoaders for CV workflows."""

    def __init__(
        self,
        data_dir: str | Path = "data/raw",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize CVDataModule.

        Args:
            data_dir: Root directory of dataset.
            batch_size: Mini-batch size for training and evaluation.
            num_workers: Number of DataLoader subprocesses.
            pin_memory: If True, copy Tensors into CUDA pinned memory.
            image_size: Target image dimensions (H, W).
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.image_size = image_size

        self.train_dataset: StandardCVDataset | None = None
        self.val_dataset: StandardCVDataset | None = None
        self.test_dataset: StandardCVDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets and transforms for specified stage."""
        train_transforms = get_base_transforms(self.image_size, is_training=True)
        eval_transforms = get_base_transforms(self.image_size, is_training=False)

        # Scaffolding mock samples if data folder is not populated yet
        dummy_samples = [("dummy.jpg", 0)]

        self.train_dataset = StandardCVDataset(dummy_samples, transform=train_transforms)
        self.val_dataset = StandardCVDataset(dummy_samples, transform=eval_transforms)
        self.test_dataset = StandardCVDataset(dummy_samples, transform=eval_transforms)

    def train_dataloader(self) -> Any:
        """Build training DataLoader."""
        from torch.utils.data import DataLoader

        if self.train_dataset is None:
            self.setup("fit")

        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Any:
        """Build validation DataLoader."""
        from torch.utils.data import DataLoader

        if self.val_dataset is None:
            self.setup("fit")

        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Any:
        """Build testing DataLoader."""
        from torch.utils.data import DataLoader

        if self.test_dataset is None:
            self.setup("test")

        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
