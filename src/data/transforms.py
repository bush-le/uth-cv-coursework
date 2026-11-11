"""Image transformation and data augmentation pipelines."""

from typing import Any, Callable, Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F

_CASCADE_CLASSIFIER = None


def get_face_cascade() -> cv2.CascadeClassifier:
    """Get or lazily initialize the shared Haar Cascade classifier for face detection."""
    global _CASCADE_CLASSIFIER
    if _CASCADE_CLASSIFIER is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE_CLASSIFIER = cv2.CascadeClassifier(cascade_path)
    return _CASCADE_CLASSIFIER


def crop_face_tensor(
    image: torch.Tensor,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    target_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Crop the facial region from an image tensor and resize to target_size.

    Args:
        image: Tensor of shape (C, H, W) in [0, 1].
        bbox: Optional normalized (xmin, ymin, xmax, ymax) coordinates.
        target_size: Desired (height, width) of output face crop.

    Returns:
        Cropped and resized face tensor of shape (C, target_size[0], target_size[1]).
    """
    c, h, w = image.shape
    if bbox is None:
        np_img = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        cascade = get_face_cascade()
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            xmin, ymin, xmax, ymax = float(fx / w), float(fy / h), float((fx + fw) / w), float((fy + fh) / h)
        else:
            xmin, ymin, xmax, ymax = (0.15, 0.15, 0.85, 0.85)
    else:
        xmin, ymin, xmax, ymax = bbox

    face_h = ymax - ymin
    face_w = xmax - xmin
    ymin = max(0.0, ymin - 0.10 * face_h)
    ymax = min(1.0, ymax + 0.10 * face_h)
    xmin = max(0.0, xmin - 0.10 * face_w)
    xmax = min(1.0, xmax + 0.10 * face_w)

    py_min, py_max = int(ymin * h), max(int(ymin * h) + 1, int(ymax * h))
    px_min, px_max = int(xmin * w), max(int(xmin * w) + 1, int(xmax * w))

    face_crop = image[:, py_min:py_max, px_min:px_max]
    face_crop_batched = face_crop.unsqueeze(0)
    resized = F.interpolate(face_crop_batched, size=target_size, mode="bilinear", align_corners=False)
    return resized.squeeze(0)


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
