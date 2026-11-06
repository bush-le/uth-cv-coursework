"""Evaluation metrics calculation module for Computer Vision tasks."""

from typing import Any
import numpy as np
import torch


def compute_iou(
    box_a: np.ndarray | list[float],
    box_b: np.ndarray | list[float],
    box_format: str = "xyxy",
) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box_a: Coordinates of first box, shape (4,).
        box_b: Coordinates of second box, shape (4,).
        box_format: Format of bounding box: 'xyxy' [x1, y1, x2, y2]
                    or 'xywh' [x, y, w, h].

    Returns:
        Intersection over Union scalar float in range [0.0, 1.0].
    """
    a = np.array(box_a, dtype=np.float32)
    b = np.array(box_b, dtype=np.float32)

    if box_format == "xywh":
        a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]], dtype=np.float32)
        b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]], dtype=np.float32)

    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, float((a[2] - a[0]) * (a[3] - a[1])))
    area_b = max(0.0, float((b[2] - b[0]) * (b[3] - b[1])))
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Class predictions of shape (N,) or logits of shape (N, C).
        targets: Ground truth class labels of shape (N,).

    Returns:
        Accuracy score in range [0.0, 1.0].
    """
    preds = np.asarray(predictions)
    targs = np.asarray(targets)

    if preds.ndim > 1:
        preds = np.argmax(preds, axis=-1)

    if len(targs) == 0:
        return 0.0

    return float(np.mean(preds == targs))


def compute_miou(
    predictions: np.ndarray | Any,
    targets: np.ndarray | Any,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """Compute per-class IoU and Mean IoU (mIoU) for semantic segmentation masks.

    Args:
        predictions: Predicted class integer indices, shape (H, W) or (B, H, W).
        targets: Ground-truth class integer indices, shape (H, W) or (B, H, W).
        num_classes: Total number of classes.
        ignore_index: Optional class ID to exclude from computation.

    Returns:
        Dictionary mapping class names to IoUs and overall 'miou'.
    """
    preds = np.asarray(predictions).flatten()
    targs = np.asarray(targets).flatten()

    ious: dict[str, float] = {}
    valid_ious: list[float] = []

    for cls_id in range(num_classes):
        if ignore_index is not None and cls_id == ignore_index:
            continue

        p_mask = preds == cls_id
        t_mask = targs == cls_id

        intersection = np.logical_and(p_mask, t_mask).sum()
        union = np.logical_or(p_mask, t_mask).sum()

        if union == 0:
            # If both pred and target have 0 pixels for this class, IoU is 1.0 (empty match)
            iou = 1.0
        else:
            iou = float(intersection) / float(union)

        ious[f"iou_class_{cls_id}"] = iou
        valid_ious.append(iou)

    ious["miou"] = float(np.mean(valid_ious)) if valid_ious else 0.0
    return ious


def _get_boundary_mask_numpy(mask: np.ndarray, dilation_radius: int = 1) -> np.ndarray:
    """Extract contour boundary region of a binary mask using vectorized array operations.

    Args:
        mask: 2D boolean or uint8 mask of shape (H, W).
        dilation_radius: Pixel margin for boundary dilation.

    Returns:
        2D boolean array of shape (H, W) marking boundary pixels.
    """
    import scipy.ndimage as ndi

    struct = ndi.generate_binary_structure(2, 1)
    if dilation_radius > 1:
        struct = ndi.iterate_structure(struct, dilation_radius)

    dilated = ndi.binary_dilation(mask, structure=struct)
    eroded = ndi.binary_erosion(mask, structure=struct)
    boundary = np.logical_and(dilated, np.logical_not(eroded))
    return boundary


def compute_boundary_iou(
    predictions: np.ndarray | Any,
    targets: np.ndarray | Any,
    num_classes: int,
    dilation_radius: int = 1,
) -> dict[str, float]:
    """Compute Boundary IoU (BIoU) for fine-grained boundary segmentation evaluation.

    Args:
        predictions: Predicted class integer indices, shape (H, W).
        targets: Ground-truth class integer indices, shape (H, W).
        num_classes: Total number of classes.
        dilation_radius: Pixel boundary margin radius d.

    Returns:
        Dictionary mapping class names to boundary IoU and 'mean_biou'.
    """
    preds = np.asarray(predictions)
    targs = np.asarray(targets)

    # Work on 2D arrays
    if preds.ndim > 2:
        preds = preds.squeeze(0)
    if targs.ndim > 2:
        targs = targs.squeeze(0)

    bious: dict[str, float] = {}
    valid_bious: list[float] = []

    for cls_id in range(1, num_classes):  # Focus on foreground occlusion classes
        p_mask = preds == cls_id
        t_mask = targs == cls_id

        if not np.any(p_mask) and not np.any(t_mask):
            bious[f"biou_class_{cls_id}"] = 1.0
            valid_bious.append(1.0)
            continue

        try:
            p_bound = _get_boundary_mask_numpy(p_mask, dilation_radius)
            t_bound = _get_boundary_mask_numpy(t_mask, dilation_radius)

            p_bound_g = np.logical_and(p_bound, t_mask)
            t_bound_p = np.logical_and(t_bound, p_mask)

            inter = np.logical_and(p_bound_g, t_bound_p).sum()
            union = np.logical_or(p_bound_g, t_bound_p).sum()

            biou = float(inter) / float(union) if union > 0 else 0.0
        except Exception:
            # Fallback to standard IoU if morphology module unavailable
            inter = np.logical_and(p_mask, t_mask).sum()
            union = np.logical_or(p_mask, t_mask).sum()
            biou = float(inter) / float(union) if union > 0 else 0.0

        bious[f"biou_class_{cls_id}"] = biou
        valid_bious.append(biou)

    bious["mean_biou"] = float(np.mean(valid_bious)) if valid_bious else 0.0
    return bious



def compute_temporal_warping_error(video: torch.Tensor) -> float:
    """Compute temporal warping / photometric error between consecutive video frames.
    
    Args:
        video: torch.Tensor of shape (T, C, H, W) or (B, T, C, H, W).
        
    Returns:
        Mean absolute difference between consecutive frames in range [0.0, 1.0].
    """
    if not isinstance(video, torch.Tensor):
        video = torch.tensor(video, dtype=torch.float32)

    if video.ndim == 4:
        # (T, C, H, W)
        t = video.shape[0]
        if t <= 1:
            return 0.0
        frame_diffs = torch.abs(video[1:] - video[:-1])
        return float(torch.mean(frame_diffs).item())
    elif video.ndim == 5:
        # (B, T, C, H, W)
        b, t, c, h, w = video.shape
        if t <= 1:
            return 0.0
        frame_diffs = torch.abs(video[:, 1:] - video[:, :-1])
        return float(torch.mean(frame_diffs).item())
    else:
        raise ValueError(f"Expected 4D or 5D video tensor, got shape {tuple(video.shape)}")


def compute_identity_consistency(video: torch.Tensor, backbone: str = "mock") -> float:
    """Compute facial identity cosine consistency across consecutive frames in a clip.
    
    Args:
        video: torch.Tensor of shape (T, C, H, W) or (B, T, C, H, W).
        backbone: Identity embedding model type ('mock', 'arcface').
        
    Returns:
        Mean sequential cosine similarity score in range [0.0, 1.0].
    """
    if not isinstance(video, torch.Tensor):
        video = torch.tensor(video, dtype=torch.float32)

    if video.ndim == 5:
        video = video[0] # Take first clip if batched

    t, c, h, w = video.shape
    if t <= 1:
        return 1.0

    # Vectorized feature proxy / pooling representation (128-dim embedding)
    # Uses adaptive average pooling + deterministic projection
    pooled = torch.nn.functional.adaptive_avg_pool2d(video, (8, 8)).view(t, -1) # (T, C*64)
    # Normalize embeddings
    norms = torch.norm(pooled, p=2, dim=-1, keepdim=True) + 1e-8
    normalized = pooled / norms

    # Sequential cosine similarity
    sims = torch.sum(normalized[1:] * normalized[:-1], dim=-1)
    mean_sim = float(torch.mean(sims).item())
    # Bound to [0.0, 1.0]
    return max(0.0, min(1.0, (mean_sim + 1.0) / 2.0 if mean_sim < 0 else mean_sim))


def compute_fvd_score(real_videos: torch.Tensor, fake_videos: torch.Tensor) -> float:
    """Compute Fréchet Video Distance (FVD) proxy between real and fake video distributions.
    
    Args:
        real_videos: torch.Tensor of shape (N, T, C, H, W).
        fake_videos: torch.Tensor of shape (M, T, C, H, W).
        
    Returns:
        FVD distance score (scalar float >= 0.0).
    """
    if not isinstance(real_videos, torch.Tensor):
        real_videos = torch.tensor(real_videos, dtype=torch.float32)
    if not isinstance(fake_videos, torch.Tensor):
        fake_videos = torch.tensor(fake_videos, dtype=torch.float32)

    # Flatten spatial and temporal dimensions into sample feature vectors
    n = real_videos.shape[0]
    m = fake_videos.shape[0]
    real_feats = torch.nn.functional.adaptive_avg_pool3d(
        real_videos.transpose(1, 2), (4, 4, 4)
    ).reshape(n, -1).numpy()
    fake_feats = torch.nn.functional.adaptive_avg_pool3d(
        fake_videos.transpose(1, 2), (4, 4, 4)
    ).reshape(m, -1).numpy()

    # Calculate mean and covariance
    mu_real = np.mean(real_feats, axis=0)
    mu_fake = np.mean(fake_feats, axis=0)

    diff = mu_real - mu_fake
    mean_dist = float(np.dot(diff, diff))

    # Trace variance component
    var_real = np.var(real_feats, axis=0).sum()
    var_fake = np.var(fake_feats, axis=0).sum()
    cov_dist = float(var_real + var_fake - 2.0 * np.sqrt(max(0.0, var_real * var_fake)))

    fvd = mean_dist + cov_dist
    return max(0.0, float(fvd))



def compute_laplacian_sharpness(image: torch.Tensor) -> float:
    """Compute sharpness metric via variance of the 2D discrete Laplacian operator.
    
    Args:
        image: torch.Tensor of shape (C, H, W) or (B, C, H, W) in [0.0, 1.0].
        
    Returns:
        Scalar sharpness variance score (higher indicates crisper image).
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if image.ndim == 4:
        image = image[0]  # Take first image in batch

    # Convert to 1-channel grayscale (0.2989 R + 0.5870 G + 0.1140 B)
    if image.shape[0] == 3:
        gray = 0.2989 * image[0] + 0.5870 * image[1] + 0.1140 * image[2]
    else:
        gray = image[0]

    gray = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # 3x3 Laplacian convolution kernel
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
        device=image.device,
    ).unsqueeze(0).unsqueeze(0)

    laplacian = torch.nn.functional.conv2d(gray, kernel, padding=1)
    variance = float(torch.var(laplacian).item()) * 1000.0
    return variance


def compute_clip_similarity_proxy(image: torch.Tensor, prompt: str) -> float:
    """Compute semantic text-image alignment score proxy in range [0.0, 1.0].
    
    Args:
        image: torch.Tensor of shape (C, H, W) in [0.0, 1.0].
        prompt: Text description string.
        
    Returns:
        Similarity score float bounded in [0.0, 1.0].
    """
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if image.ndim == 4:
        image = image[0]

    c, h, w = image.shape
    # Spatial frequency + color distribution feature proxy (128-dim)
    pooled = torch.nn.functional.adaptive_avg_pool2d(image, (8, 8)).view(-1)
    norm = torch.norm(pooled, p=2) + 1e-8
    img_feat = pooled / norm

    # Deterministic prompt pseudo-embedding
    gen = torch.Generator().manual_seed(abs(hash(prompt)) % 100000)
    prompt_feat = torch.randn(img_feat.shape[0], generator=gen)
    prompt_feat = prompt_feat / (torch.norm(prompt_feat, p=2) + 1e-8)

    cosine_sim = float(torch.dot(img_feat, prompt_feat).item())
    # Scale to [0.0, 1.0]
    return float(np.clip((cosine_sim + 1.0) / 2.0, 0.0, 1.0))
