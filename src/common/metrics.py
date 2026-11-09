"""Evaluation and quality curation metrics for 2D Deepfake Detection and Hard-Sample Mining."""

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from src.common.logger import get_logger

logger = get_logger(__name__)


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


def compute_precision(
    predictions: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    positive_label: int = 1,
) -> float:
    """Compute binary classification precision for target positive label."""
    preds = np.asarray(predictions).ravel()
    targs = np.asarray(targets).ravel()
    tp = int(np.sum((preds == positive_label) & (targs == positive_label)))
    fp = int(np.sum((preds == positive_label) & (targs != positive_label)))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def compute_recall(
    predictions: np.ndarray | list[int],
    targets: np.ndarray | list[int],
    positive_label: int = 1,
) -> float:
    """Compute binary classification recall for target positive label."""
    preds = np.asarray(predictions).ravel()
    targs = np.asarray(targets).ravel()
    tp = int(np.sum((preds == positive_label) & (targs == positive_label)))
    fn = int(np.sum((preds != positive_label) & (targs == positive_label)))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))



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


def calibrate_threshold(reference_scores: Sequence[float], quantile: float) -> float:
    """Calibrate a decision threshold as a quantile over a reference distribution.

    Args:
        reference_scores: Sequence of scores from reference distribution.
        quantile: Desired quantile level in [0.0, 1.0].

    Returns:
        Scalar threshold float.
    """
    if len(reference_scores) == 0:
        raise ValueError("reference_scores cannot be empty for calibration.")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0.0, 1.0], got {quantile}")
    arr = np.asarray(reference_scores, dtype=np.float64)
    return float(np.quantile(arr, quantile))


def detect_faces(image_rgb: Union[np.ndarray, torch.Tensor], backend: str = "opencv_haar") -> int:
    """Detect faces in an RGB uint8 image using classical or lightweight detector.

    Args:
        image_rgb: NumPy array of shape (H, W, 3) in uint8 RGB format or torch.Tensor.
        backend: Face detector backend ('opencv_haar' or 'none').

    Returns:
        Number of faces detected.
    """
    if backend == "none":
        return 1

    if backend in ("opencv", "opencv_haar"):
        import cv2

        if isinstance(image_rgb, torch.Tensor):
            t = image_rgb.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t.permute(1, 2, 0)
            if t.dtype in (torch.float32, torch.float64):
                t = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            image_rgb = t.numpy()

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            logger.warning("Haarcascade file could not be loaded; defaulting to 1 face.")
            return 1

        if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb * 255.0 if np.issubdtype(image_rgb.dtype, np.floating) else image_rgb, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            # Relaxed pass for slightly tilted faces or sunglasses
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=2,
                minSize=(40, 40),
            )
        if len(faces) <= 1:
            return len(faces)

        # Filter out minor background texture false alarms if dominant face exists
        boxes = sorted([tuple(b) for b in faces], key=lambda b: b[2] * b[3], reverse=True)
        dominant_area = boxes[0][2] * boxes[0][3]
        sig_faces = [b for b in boxes if (b[2] * b[3]) >= 0.20 * dominant_area]
        return len(sig_faces)

    raise ValueError(f"Unknown face detection backend: {backend}")


def verify_facial_geometry(
    image: Union[np.ndarray, torch.Tensor],
) -> Tuple[bool, List[str]]:
    """Verify biometric facial geometry and reject anatomical anomalies (e.g. dual mouths, distorted proportions).

    Args:
        image: Image array (H, W, 3) in uint8 or float, or torch.Tensor (C, H, W).

    Returns:
        (is_valid, list of rejection reasons).
    """
    import cv2

    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        if t.dtype in (torch.float32, torch.float64):
            t = (t.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        image = t.numpy()

    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        image = np.clip(
            image * 255.0 if np.issubdtype(image.dtype, np.floating) else image, 0, 255
        ).astype(np.uint8)

    reasons: List[str] = []
    h, w = image.shape[:2]
    if (h, w) != (512, 512):
        image = cv2.resize(image, (512, 512))
        h, w = 512, 512

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        # Fallback to relaxed detection for tilted faces or dark eyeglasses
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(40, 40))

    if len(faces) == 0:
        return False, ["No face landmark detected"]

    # Filter out spurious Haar noise boxes (e.g. tiny patches on clothing/textures)
    # when a prominent dominant portrait face exists
    faces_list = [tuple(b) for b in faces]
    faces_list.sort(key=lambda b: b[2] * b[3], reverse=True)
    dominant_box = faces_list[0]
    dom_area = dominant_box[2] * dominant_box[3]
    total_area = float(w * h)

    # Significant faces must be at least 20% of the dominant face area and at least 5% of total frame
    significant_faces = [
        b for b in faces_list
        if (b[2] * b[3]) >= 0.20 * dom_area and (b[2] * b[3]) >= 0.05 * total_area
    ]

    if len(significant_faces) > 1:
        return False, [f"Multiple faces detected ({len(significant_faces)}), expecting single portrait"]

    fx, fy, fw, fh = dominant_box
    aspect_ratio = fh / max(1, fw)
    if aspect_ratio < 0.65 or aspect_ratio > 1.6:
        reasons.append(f"Abnormal facial aspect ratio: {aspect_ratio:.2f}")

    # Area ratio check (face must occupy 20% to 90% of frame, rejecting miniature mirror faces and extreme crops)
    face_area_ratio = (fw * fh) / float(w * h)
    if face_area_ratio < 0.20:
        reasons.append(f"Face occupies too small area of frame ({face_area_ratio:.2%} < 20%): likely vignette/distant frame")
    elif face_area_ratio > 0.90:
        reasons.append(f"Face occupies too large area of frame ({face_area_ratio:.2%} > 90%): extreme crop")

    # Centering check
    center_x = (fx + fw / 2.0) / float(w)
    center_y = (fy + fh / 2.0) / float(h)
    if center_x < 0.15 or center_x > 0.85 or center_y < 0.15 or center_y > 0.85:
        reasons.append(f"Face is severely off-center: ({center_x:.2f}, {center_y:.2f})")

    # Inspect lower face for horizontal inpainting boundary / seam spike
    lower = gray[fy + int(0.35 * fh) : fy + int(0.85 * fh), fx + int(0.1 * fw) : fx + int(0.9 * fw)]
    if lower.shape[0] >= 10 and lower.shape[1] >= 10:
        sobel_y = np.abs(cv2.Sobel(lower, cv2.CV_64F, 0, 1, ksize=3))
        row_means = np.mean(sobel_y, axis=1)
        peak = float(np.max(row_means))
        median = float(np.median(row_means)) + 1e-5
        spike = peak / median
        # Real mouths and teeth have spike ~5.0-6.5 and peak < 20. Artificial horizontal seams have spike > 8.0 AND peak > 20.0
        if spike > 8.0 and peak > 20.0:
            reasons.append(f"Severe horizontal boundary discontinuity across face (seam_spike={spike:.2f} > 8.0, peak={peak:.1f})")

    is_valid = len(reasons) == 0
    return is_valid, reasons


def verify_boundary_seam(
    synthetic: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    max_gradient_diff: float = 35.0,
) -> Tuple[bool, float]:
    """Verify that inpainting boundaries blend seamlessly without abrupt seam lines.

    Args:
        synthetic: Generated image tensor (C, H, W) in [0, 1].
        source: Source image tensor (C, H, W) in [0, 1].
        mask: Inpainting mask tensor (1, H, W) in [0, 1].
        max_gradient_diff: Maximum acceptable boundary gradient difference.

    Returns:
        (is_seamless, boundary_discontinuity_score).
    """
    import cv2

    mask_np = (mask[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask_np, kernel, iterations=1)
    eroded = cv2.erode(mask_np, kernel, iterations=1)
    boundary_ring = (dilated > 50) & (eroded < 200)

    if not np.any(boundary_ring):
        return True, 0.0

    syn_np = (synthetic.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.float32)
    src_np = (source.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.float32)

    diff = np.abs(syn_np - src_np)
    boundary_diff = float(np.mean(diff[boundary_ring]))
    is_seamless = boundary_diff <= max_gradient_diff
    return is_seamless, boundary_diff


class ClipScorer:
    """Real CLIP semantic image-text similarity evaluator using Hugging Face transformers."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from transformers import CLIPModel, CLIPProcessor

            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()

    def score(self, image: torch.Tensor, prompt: str) -> float:
        """Compute cosine similarity between image (3, H, W) and prompt text."""
        self._ensure_loaded()
        from PIL import Image

        img_tensor = image.detach().cpu().clamp(0.0, 1.0)
        np_arr = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(np_arr)

        inputs = self._processor(
            text=[prompt],
            images=pil_img,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            img_feats = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_feats = self._model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )

            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            sim = (img_feats * txt_feats).sum(dim=-1).item()

        return float(np.clip(sim, -1.0, 1.0))
