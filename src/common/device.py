"""Hardware device detection and memory optimization utilities for deep learning runtimes."""

from typing import Any, Optional
import torch

from src.common.logger import get_logger

logger = get_logger(__name__)


def get_optimal_device(preferred_device: Optional[str] = None) -> torch.device:
    """Detect and return the optimal compute device (CUDA GPU or CPU).

    Args:
        preferred_device: Optional user preference ('cuda' or 'cpu').

    Returns:
        torch.device configured for execution.
    """
    if preferred_device:
        pref = preferred_device.lower()
        if pref.startswith("cuda") and torch.cuda.is_available():
            return torch.device(preferred_device)
        if pref.startswith("cpu"):
            return torch.device("cpu")

    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        logger.info("CUDA device detected: %s", dev_name)
        return torch.device("cuda:0")

    logger.info("No CUDA GPU detected or PyTorch built without CUDA, using CPU.")
    return torch.device("cpu")


def get_torch_dtype(device: Optional[torch.device] = None) -> torch.dtype:
    """Return appropriate floating point dtype depending on device capability.

    Args:
        device: Target torch device.

    Returns:
        torch.float16 for CUDA, torch.float32 for CPU.
    """
    if device is None:
        device = get_optimal_device()

    if device.type == "cuda":
        return torch.float16
    return torch.float32


def optimize_pipeline_memory(pipeline: Any) -> None:
    """Apply memory reduction strategies to Diffusers diffusion pipelines.

    Enables attention slicing and VAE slicing to safely run within limited VRAM (e.g. 4GB).

    Args:
        pipeline: Hugging Face Diffusers pipeline instance or compatible object.
    """
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing("auto")
        logger.info("Enabled attention slicing on diffusion pipeline.")

    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        logger.info("Enabled VAE slicing on diffusion pipeline.")
