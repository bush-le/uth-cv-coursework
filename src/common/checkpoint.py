"""Full-state checkpointing and exact resume utilities adhering to project governance."""

from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.common.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    state: dict[str, Any],
    is_best: bool,
    checkpoint_dir: str | Path,
    run_name: str,
) -> tuple[Path, Path | None]:
    """Save full-state checkpoint for resume and best model tracking.

    Args:
        state: State dictionary containing model, optimizer, epoch, etc.
        is_best: Whether this checkpoint achieved the best validation metric.
        checkpoint_dir: Directory where checkpoints should be stored.
        run_name: Name identifier of current run.

    Returns:
        Tuple of (path_to_last_checkpoint, path_to_best_checkpoint_or_None).
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # Capture RNG states if not already provided
    if "rng_state" not in state:
        state["rng_state"] = torch.get_rng_state()
    if torch.cuda.is_available() and "cuda_rng_state" not in state:
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    last_pt = ckpt_path / f"{run_name}_last.pt"
    torch.save(state, last_pt)
    logger.debug("Saved latest state checkpoint to: %s", last_pt)

    best_pt: Path | None = None
    if is_best:
        best_pt = ckpt_path / f"{run_name}_best.pt"
        torch.save(state, best_pt)
        logger.info("New best model checkpoint recorded at: %s", best_pt)

    return last_pt, best_pt


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Load and restore full training state from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model: nn.Module to load weights into.
        optimizer: Optional optimizer to restore states into.
        scheduler: Optional learning rate scheduler to restore.
        device: Target execution device.

    Returns:
        The loaded checkpoint dictionary.
    """
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    logger.info("Restoring training state from checkpoint: %s", ckpt_file)
    checkpoint = torch.load(ckpt_file, map_location=device)

    # Restore model parameters
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Restore optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore random number generator state
    if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
        try:
            torch.set_rng_state(checkpoint["rng_state"])
        except Exception as e:
            logger.warning("Could not restore torch RNG state: %s", e)

    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint and checkpoint["cuda_rng_state"] is not None:
        try:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        except Exception as e:
            logger.warning("Could not restore CUDA RNG state: %s", e)

    logger.info(
        "Successfully restored checkpoint from epoch %s (best metric: %s)",
        checkpoint.get("epoch", "unknown"),
        checkpoint.get("best_metric", "unknown"),
    )
    return checkpoint
