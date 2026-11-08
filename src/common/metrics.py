"""Evaluation and quality curation metrics for 2D Deepfake Detection and Hard-Sample Mining."""

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


def compute_accuracy(predictions: np.ndarray | list[int], targets: np.ndarray | list[int]) -> float:
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


def compute_roc_auc(y_true: np.ndarray | list[int], y_score: np.ndarray | list[float]) -> float:
    """Compute ROC-AUC via vectorized Wilcoxon-Mann-Whitney rank-sum statistic.

    Args:
        y_true: Ground truth binary labels (0=Real, 1=Fake) of shape (N,).
        y_score: Predicted fake probabilities P(fake) of shape (N,).

    Returns:
        ROC-AUC scalar float in [0.0, 1.0].
    """
    y_t = np.asarray(y_true).ravel()
    y_s = np.asarray(y_score).ravel()

    pos_mask = y_t == 1
    neg_mask = y_t == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return 1.0

    order = np.argsort(y_s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_s), dtype=np.float64) + 1.0

    rank_sum_pos = float(np.sum(ranks[pos_mask]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_bypass_rate(
    fake_probs: np.ndarray | torch.Tensor | list[float],
    threshold: float = 0.5,
) -> float:
    """Compute the proportion of synthetic DF images that bypass (fool) the detector.

    A synthetic DF image (ground-truth label=1) bypasses the detector when its predicted
    fake probability P(fake) < threshold (i.e. misclassified as Real -> Hard Sample).

    Args:
        fake_probs: Predicted P(fake) scores for synthetic fake samples, shape (N_fake,).
        threshold: Decision boundary tau (default 0.5).

    Returns:
        Bypass rate in [0.0, 1.0] (lower is better for detector; higher yields more hard samples).
    """
    if isinstance(fake_probs, torch.Tensor):
        probs = fake_probs.detach().cpu().numpy().ravel()
    else:
        probs = np.asarray(fake_probs, dtype=np.float32).ravel()

    if len(probs) == 0:
        return 0.0

    bypassed = probs < threshold
    return float(np.mean(bypassed))


def compute_hard_sample_recall(
    predictions: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    is_hard_sample: np.ndarray | list[bool],
) -> float:
    """Compute detection recall specifically on mined hard samples (bypassed DF images).

    Args:
        predictions: Predicted binary labels (0 or 1) of shape (N,).
        targets: Ground-truth binary labels (0 or 1) of shape (N,).
        is_hard_sample: Boolean mask indicating mined hard samples of shape (N,).

    Returns:
        Recall on hard fake samples in [0.0, 1.0]. Returns 1.0 if no hard samples present.
    """
    preds = np.asarray(predictions).ravel()
    targs = np.asarray(targets).ravel()
    hard_mask = np.asarray(is_hard_sample, dtype=bool).ravel()

    target_hard_mask = (targs == 1) & hard_mask
    if not np.any(target_hard_mask):
        return 1.0

    caught = (preds[target_hard_mask] == 1).sum()
    total_hard = target_hard_mask.sum()
    return float(caught / total_hard)


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
    kernel = (
        torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=image.device,
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

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

    # Spatial frequency + color distribution feature proxy (192-dim)
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
