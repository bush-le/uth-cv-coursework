"""Unit tests for full-state checkpoint save and resume utilities."""

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.common.checkpoint import load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint(tmp_path):
    model = nn.Linear(10, 2)
    optimizer = AdamW(model.parameters(), lr=0.01)

    state = {
        "epoch": 5,
        "global_step": 120,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": 0.85,
        "history": {"val_miou": [0.70, 0.75, 0.80, 0.82, 0.85]},
    }

    last_pt, best_pt = save_checkpoint(
        state=state,
        is_best=True,
        checkpoint_dir=tmp_path,
        run_name="test_run",
    )

    assert last_pt.exists()
    assert best_pt is not None and best_pt.exists()

    # Create fresh model and optimizer to verify exact restore
    restored_model = nn.Linear(10, 2)
    restored_optimizer = AdamW(restored_model.parameters(), lr=0.01)

    restored_state = load_checkpoint(
        checkpoint_path=last_pt,
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
    )

    assert restored_state["epoch"] == 5
    assert restored_state["global_step"] == 120
    assert restored_state["best_metric"] == pytest.approx(0.85)

    # Check that weights match
    for p1, p2 in zip(model.parameters(), restored_model.parameters()):
        assert torch.allclose(p1, p2)
