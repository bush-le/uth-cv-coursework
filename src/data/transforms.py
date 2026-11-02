"""Image transformation and data augmentation pipelines."""

from typing import Any, Callable
import numpy as np


def normalize_image(
    image: np.ndarray,
    mean: list[float] | tuple[float, ...] = (0.485, 0.456, 0.406),
    std: list[float] | tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize image array with mean and standard deviation.

    Args:
        image: Floating point image in range [0.0, 1.0] of shape (H, W, C) or (C, H, W).
        mean: Channel means.
        std: Channel standard deviations.

    Returns:
        Normalized image array of same shape.
    """
    img = np.asarray(image, dtype=np.float32)
    m = np.array(mean, dtype=np.float32)
    s = np.array(std, dtype=np.float32)

    if img.ndim == 3 and img.shape[0] == len(mean): # (C, H, W)
        m = m[:, None, None]
        s = s[:, None, None]
    elif img.ndim == 3 and img.shape[2] == len(mean): # (H, W, C)
        m = m[None, None, :]
        s = s[None, None, :]

    return (img - m) / s


def get_base_transforms(
    image_size: tuple[int, int] = (224, 224),
    is_training: bool = False,
) -> Callable[[Any], Any]:
    """Get standardized torchvision transforms for training or evaluation.

    Args:
        image_size: Target tuple (H, W).
        is_training: Whether to include data augmentations.

    Returns:
        Composed transform callable.
    """
    try:
        from torchvision import transforms

        transform_list = []
        transform_list.append(transforms.Resize(image_size))

        if is_training:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        return transforms.Compose(transform_list)
    except ImportError:
        def _dummy_transform(x: Any) -> Any:
            return x

        return _dummy_transform
