"""Evaluation metrics calculation module for Computer Vision tasks."""

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
