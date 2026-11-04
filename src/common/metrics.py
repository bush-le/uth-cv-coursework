"""Evaluation metrics calculation module for Computer Vision tasks."""

from typing import Any
import numpy as np


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
