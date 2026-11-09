"""Hardness evaluation and real-manifold mining for synthetic deepfake datasets."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common.logger import get_logger
from src.data.image_datamodule import collate_image_samples
from src.data.image_dataset import DeepfakeImageDataset
from src.data.image_schema import ImageMetadata

logger = get_logger(__name__)


@dataclass
class HardnessReport:
    """Standardized audit report of hardness evaluation across synthetic dataset."""

    method: str
    tau: float
    tau_source: str
    total: int
    hard_count: int
    hard_rate: float
    hard_rate_by_mode: Dict[str, float]
    hard_rate_by_group: Dict[str, Dict[str, float]]
    hard_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def embed_images(
    extractor: nn.Module,
    metas: List[ImageMetadata],
    batch_size: int = 16,
    device: Union[str, torch.device] = "cpu",
    cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Extract L2-normalized deep representations for a list of image metadata.

    Args:
        extractor: Neural feature extractor (e.g., DINOv2 backbone).
        metas: List of ImageMetadata items.
        batch_size: Batch size for forward pass.
        device: Device to execute forward pass.
        cache: Optional memory cache mapping image_id to tensor.

    Returns:
        Tensor of shape (N, D) where N = len(metas), D = embedding dimension.
    """
    if not metas:
        return torch.empty((0, 384), dtype=torch.float32)

    dev = torch.device(device)
    extractor.eval()
    extractor.to(dev)

    dataset = DeepfakeImageDataset(metadata_list=metas, cache=cache)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_image_samples,
    )

    all_embs: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(dev)
            emb = extractor(imgs)
            emb_norm = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb_norm.detach().cpu())

    if all_embs:
        return torch.cat(all_embs, dim=0)
    return torch.empty((0, 384), dtype=torch.float32)


def real_manifold_scores(
    fake_emb: torch.Tensor,
    real_emb: torch.Tensor,
    k: int = 5,
    exclude_idx: Optional[List[Optional[int]]] = None,
) -> torch.Tensor:
    """Calculate real-manifold proximity score as top-k cosine similarity to real faces.

    Args:
        fake_emb: Tensor of shape (N, D), L2-normalized.
        real_emb: Tensor of shape (M, D), L2-normalized.
        k: Number of nearest neighbors to average.
        exclude_idx: Optional list of length N specifying the index in real_emb to exclude.

    Returns:
        Tensor of shape (N,) float values representing mean top-k cosine similarity.
    """
    N, D = fake_emb.shape
    M, _ = real_emb.shape
    if N == 0 or M == 0:
        return torch.zeros(N, dtype=torch.float32)

    sim_matrix = torch.matmul(fake_emb, real_emb.T).clone()  # (N, M)

    if exclude_idx is not None:
        for i, ex in enumerate(exclude_idx):
            if ex is not None and 0 <= ex < M:
                sim_matrix[i, ex] = -float("inf")

    k_val = min(k, M)
    topk_vals, _ = torch.topk(sim_matrix, k=k_val, dim=1)
    # Mask any -inf if all were excluded
    finite_mask = torch.isfinite(topk_vals)
    counts = finite_mask.sum(dim=1).clamp(min=1)
    safe_vals = torch.where(finite_mask, topk_vals, torch.zeros_like(topk_vals))
    return safe_vals.sum(dim=1) / counts


def real_reference_scores(
    real_emb: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Leave-one-out self-similarity distribution of real reference faces.

    Args:
        real_emb: Tensor of shape (M, D), L2-normalized.
        k: Number of nearest neighbors to average.

    Returns:
        Tensor of shape (M,) float values representing self-similarity on the real manifold.
    """
    M, D = real_emb.shape
    if M <= 1:
        return torch.zeros(M, dtype=torch.float32)

    sim_matrix = torch.matmul(real_emb, real_emb.T).clone()  # (M, M)
    diag_indices = torch.arange(M)
    sim_matrix[diag_indices, diag_indices] = -float("inf")

    k_val = min(k, max(1, M - 1))
    topk_vals, _ = torch.topk(sim_matrix, k=k_val, dim=1)
    return topk_vals.mean(dim=1)


def load_verified_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> None:
    """Load model checkpoint and strictly verify classification head integrity.

    Zero-Mock Invariant: Must fail if model_state_dict is missing or if classifier head weights
    are absent, preventing silent execution with random weights.

    Args:
        model: PyTorch model whose state should be loaded.
        path: Path to checkpoint .pt file.
        device: Target compute device.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format is invalid or classifier head weights are missing.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")

    checkpoint = torch.load(p, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint {p} does not contain required 'model_state_dict' dictionary.")

    sd = checkpoint["model_state_dict"]
    model_keys = list(model.state_dict().keys())
    head_keys = [k for k in model_keys if k.startswith("classifier") or k.startswith("head")]

    missing_head = [k for k in head_keys if k not in sd]
    if missing_head:
        raise ValueError(
            f"Checkpoint {p} is missing essential classification head weights: {missing_head}. "
            f"See .agents/rules/ZERO_MOCK_DATA_RULE.md"
        )

    model.load_state_dict(sd, strict=False)


def build_report(
    metas: List[ImageMetadata],
    scores: torch.Tensor,
    tau: float,
    tau_source: str,
    method: str,
    higher_is_harder: bool = True,
) -> HardnessReport:
    """Construct structured HardnessReport analyzing results by manipulation mode and demographics.

    Args:
        metas: List of synthetic ImageMetadata items.
        scores: Float tensor of shape (N,) containing hardness scores.
        tau: Decision threshold separating hard samples from standard synthetic items.
        tau_source: Explanation or origin of tau (e.g. 'calibrated_quantile', 'config').
        method: Hardness evaluation method ('real_manifold' or 'detector').
        higher_is_harder: True if score >= tau flags hard sample, False if score < tau flags hard sample.

    Returns:
        Structured HardnessReport.
    """
    total = len(metas)
    if total == 0:
        return HardnessReport(
            method=method,
            tau=tau,
            tau_source=tau_source,
            total=0,
            hard_count=0,
            hard_rate=0.0,
            hard_rate_by_mode={},
            hard_rate_by_group={},
            hard_ids=[],
        )

    scores_list = scores.detach().cpu().tolist()
    hard_ids: List[str] = []
    mode_counts: Dict[str, int] = {}
    mode_hard: Dict[str, int] = {}

    group_counts: Dict[str, Dict[str, int]] = {"gender": {}, "age_group": {}}
    group_hard: Dict[str, Dict[str, int]] = {"gender": {}, "age_group": {}}

    for meta, s in zip(metas, scores_list):
        mode = meta.manipulation_type or "unknown"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

        demo = meta.demographics or {}
        g = demo.get("gender", "unknown")
        a = demo.get("age_group", "unknown")

        group_counts["gender"][g] = group_counts["gender"].get(g, 0) + 1
        group_counts["age_group"][a] = group_counts["age_group"].get(a, 0) + 1

        is_hard = (s >= tau) if higher_is_harder else (s < tau)
        if is_hard:
            hard_ids.append(meta.image_id)
            mode_hard[mode] = mode_hard.get(mode, 0) + 1
            group_hard["gender"][g] = group_hard["gender"].get(g, 0) + 1
            group_hard["age_group"][a] = group_hard["age_group"].get(a, 0) + 1

    hard_rate_by_mode = {
        m: float(mode_hard.get(m, 0) / count) for m, count in mode_counts.items()
    }

    hard_rate_by_group: Dict[str, Dict[str, float]] = {}
    for attr, counts_map in group_counts.items():
        hard_rate_by_group[attr] = {
            val: float(group_hard[attr].get(val, 0) / count) for val, count in counts_map.items()
        }

    return HardnessReport(
        method=method,
        tau=tau,
        tau_source=tau_source,
        total=total,
        hard_count=len(hard_ids),
        hard_rate=float(len(hard_ids) / total),
        hard_rate_by_mode=hard_rate_by_mode,
        hard_rate_by_group=hard_rate_by_group,
        hard_ids=hard_ids,
    )
