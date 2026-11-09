"""Objective cross-model hardness verification without model-specific evaluation bias."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F

from src.common.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ObjectiveHardnessReport:
    """Audit report comparing objective manifold proximity across generations."""

    mean_sim_h1: float
    mean_sim_h2: float
    proximity_delta: float
    h1_count: int
    h2_count: int
    real_ref_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_manifold_proximity_gain(
    h1_embeddings: torch.Tensor,
    h2_embeddings: torch.Tensor,
    real_embeddings: torch.Tensor,
    k: int = 3,
) -> Dict[str, float]:
    """Compute average k-NN cosine proximity to reference real distribution for H1 and H2.

    Higher cosine similarity to real faces objectively proves higher visual realism
    and closer alignment to the natural human face manifold.

    Args:
        h1_embeddings: L2-normalized or unnormalized embeddings of H1 (N1, D).
        h2_embeddings: L2-normalized or unnormalized embeddings of H2 (N2, D).
        real_embeddings: Reference real face embeddings (N_real, D).
        k: Number of nearest neighbors on real manifold.

    Returns:
        Dictionary with mean_sim_h1, mean_sim_h2, and proximity_delta.
    """
    if len(h1_embeddings) == 0 or len(h2_embeddings) == 0 or len(real_embeddings) == 0:
        return {"mean_sim_h1": 0.0, "mean_sim_h2": 0.0, "proximity_delta": 0.0}

    real_norm = F.normalize(real_embeddings.float(), dim=-1)
    h1_norm = F.normalize(h1_embeddings.float(), dim=-1)
    h2_norm = F.normalize(h2_embeddings.float(), dim=-1)

    k_eff = min(k, len(real_embeddings))

    # Pairwise cosine similarity matrix: (N_syn, N_real)
    sims_h1 = torch.mm(h1_norm, real_norm.t())
    topk_h1, _ = torch.topk(sims_h1, k=k_eff, dim=-1)
    mean_sim_h1 = float(topk_h1.mean().item())

    sims_h2 = torch.mm(h2_norm, real_norm.t())
    topk_h2, _ = torch.topk(sims_h2, k=k_eff, dim=-1)
    mean_sim_h2 = float(topk_h2.mean().item())

    delta = mean_sim_h2 - mean_sim_h1

    logger.info(
        "Objective Manifold Proximity: H1=%.4f | H2=%.4f | Gain=%.4f",
        mean_sim_h1,
        mean_sim_h2,
        delta,
    )
    return {
        "mean_sim_h1": mean_sim_h1,
        "mean_sim_h2": mean_sim_h2,
        "proximity_delta": delta,
    }
