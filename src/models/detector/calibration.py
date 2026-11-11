"""Detector probability calibration via temperature scaling."""

from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim

from src.common.logger import get_logger

logger = get_logger(__name__)


class TemperatureScaler(nn.Module):
    """Post-hoc probability calibration using a single learnable temperature parameter T > 0.

    Minimizes Binary Cross-Entropy on validation logits so predicted probabilities
    accurately reflect objective true likelihoods.
    """

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by 1/T."""
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def calibrate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to calibrated probabilities in [0, 1]."""
        with torch.no_grad():
            scaled = self.forward(logits)
            return torch.sigmoid(scaled)

    def fit(
        self,
        logits: Union[torch.Tensor, list[float]],
        labels: Union[torch.Tensor, list[int]],
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Learn optimal temperature T using L-BFGS to minimize BCE loss.

        Args:
            logits: 1D tensor of raw detector logits.
            labels: 1D binary ground-truth labels (0=Real, 1=Fake).
            max_iter: Max optimization iterations.
            lr: Learning rate.

        Returns:
            Learned temperature float value.
        """
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.float32)

        logits = logits.detach().float().view(-1)
        labels = labels.detach().float().view(-1)

        bce_criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _eval() -> torch.Tensor:
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = bce_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        with torch.no_grad():
            self.temperature.clamp_(min=0.01, max=10.0)

        learned_t = float(self.temperature.item())
        logger.info("Calibrated temperature scaling: T=%.4f", learned_t)
        return learned_t
