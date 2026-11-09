"""Unit tests for objective cross-model hardness verification."""

import pytest
import torch
from src.models.evaluators.objective_hardness import compute_manifold_proximity_gain


def test_objective_manifold_density_comparison():
    """Verify compute_manifold_proximity_gain calculates positive delta when H2 is closer to real manifold than H1."""
    # Real embeddings (clustered around unit vector 0)
    real_emb = torch.tensor([[1.0, 0.0], [0.99, 0.05]])
    # H1 embeddings (average similarity ~0.8)
    h1_emb = torch.tensor([[0.8, 0.6], [0.85, 0.52]])
    # H2 embeddings (average similarity ~0.95 - closer to real)
    h2_emb = torch.tensor([[0.95, 0.31], [0.96, 0.28]])

    gain = compute_manifold_proximity_gain(h1_emb, h2_emb, real_emb)
    assert gain["mean_sim_h1"] < gain["mean_sim_h2"]
    assert gain["proximity_delta"] > 0.0
    assert 0.0 <= gain["mean_sim_h1"] <= 1.0
    assert 0.0 <= gain["mean_sim_h2"] <= 1.0
