"""DataModule managing deepfake image dataset splits, identity-disjoint partitioning, and hard-sample oversampling."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader

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
            'is_hard_sample': torch.Tensor of shape (B,) bool
            'image_ids': List[str]
            'metadata': List[ImageMetadata]
    """
    images = torch.stack([s.image for s in samples], dim=0)  # (B, 3, H, W)
    labels = torch.tensor([s.label for s in samples], dtype=torch.long)
    is_hard = torch.tensor([s.is_hard_sample for s in samples], dtype=torch.bool)
    image_ids = [s.metadata.image_id for s in samples]
    metadatas = [s.metadata for s in samples]

    return {
        "images": images,
        "labels": labels,
        "is_hard_sample": is_hard,
        "image_ids": image_ids,
        "metadata": metadatas,
    }


class DeepfakeImageDataModule:
    """DataModule managing deepfake image dataset train/val/test splits with Identity-Disjoint guarantee."""

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
        hard_sample_oversample_factor: int = 1,
        identity_disjoint: bool = True,
        seed: int = 42,
    ) -> None:
        self.metadata_list = metadata_list
        self.cache = cache
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.hard_sample_oversample_factor = max(1, hard_sample_oversample_factor)
        self.identity_disjoint = identity_disjoint
        self.seed = seed

        self.train_dataset: Optional[DeepfakeImageDataset] = None
        self.val_dataset: Optional[DeepfakeImageDataset] = None
        self.test_dataset: Optional[DeepfakeImageDataset] = None

    def _partition_splits(
        self,
    ) -> Tuple[List[ImageMetadata], List[ImageMetadata], List[ImageMetadata]]:
        """Partition metadata deterministically while keeping identity groups together."""
        total = len(self.metadata_list)
        if total == 0:
            return [], [], []

        if not self.identity_disjoint:
            train_len = int(total * self.train_ratio)
            val_len = int(total * self.val_ratio)
            generator = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(total, generator=generator).tolist()
            train_meta = [self.metadata_list[i] for i in perm[:train_len]]
            val_meta = [self.metadata_list[i] for i in perm[train_len : train_len + val_len]]
            test_meta = [self.metadata_list[i] for i in perm[train_len + val_len :]]
            return train_meta, val_meta, test_meta

        # Group by parent identity (source_real_id if available, else image_id)
        groups: Dict[str, List[ImageMetadata]] = {}
        for meta in self.metadata_list:
            key = meta.source_real_id if meta.source_real_id else meta.image_id
            groups.setdefault(key, []).append(meta)

        # Stratified Group Partitioning: separate groups containing fake samples from real-only groups
        fake_group_keys: List[str] = []
        real_group_keys: List[str] = []
        for key in sorted(groups.keys()):
            if any(m.label == 1 for m in groups[key]):
                fake_group_keys.append(key)
            else:
                real_group_keys.append(key)

        generator = torch.Generator().manual_seed(self.seed)

        def _partition_group_keys(keys: List[str]) -> Tuple[List[str], List[str], List[str]]:
            if not keys:
                return [], [], []
            perm_idx = torch.randperm(len(keys), generator=generator).tolist()
            shuffled = [keys[i] for i in perm_idx]

            # When 3+ groups exist, guarantee at least 1 in val and 1 in test
            if len(shuffled) >= 3:
                n_val = max(1, round(len(shuffled) * self.val_ratio))
                n_test = max(1, round(len(shuffled) * self.test_ratio))
                if n_val + n_test >= len(shuffled):
                    n_val = 1
                    n_test = 1
                n_train = len(shuffled) - n_val - n_test
                return shuffled[:n_train], shuffled[n_train : n_train + n_val], shuffled[n_train + n_val :]
            elif len(shuffled) == 2:
                return shuffled[:1], [], shuffled[1:]
            else:
                return shuffled, [], []

        tr_fake, val_fake, te_fake = _partition_group_keys(fake_group_keys)
        tr_real, val_real, te_real = _partition_group_keys(real_group_keys)

        train_meta: List[ImageMetadata] = []
        val_meta: List[ImageMetadata] = []
        test_meta: List[ImageMetadata] = []

        for key in tr_fake + tr_real:
            train_meta.extend(groups[key])
        for key in val_fake + val_real:
            val_meta.extend(groups[key])
        for key in te_fake + te_real:
            test_meta.extend(groups[key])

        return train_meta, val_meta, test_meta

    def setup(self) -> None:
        """Partition metadata into train, val, and test splits and apply hard-sample oversampling on train."""
        train_meta, val_meta, test_meta = self._partition_splits()

        # Oversample mined hard samples in the training split when requested
        if self.hard_sample_oversample_factor > 1 and train_meta:
            extra_hard = [
                m
                for m in train_meta
                if m.is_hard_sample
                for _ in range(self.hard_sample_oversample_factor - 1)
            ]
            train_meta = train_meta + extra_hard

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
            "ImageDataModule setup complete: %d train (oversample=%dx), %d val, %d test samples.",
            len(train_meta),
            self.hard_sample_oversample_factor,
            len(val_meta),
            len(test_meta),
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
