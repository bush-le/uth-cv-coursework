"""Unit tests for detector probability calibration using temperature scaling."""

import pytest
import torch
from src.models.detector.calibration import TemperatureScaler


def test_temperature_scaling_calibration_improves_ece():
    """Verify temperature scaler learns positive temperature and maps logits to calibrated probabilities."""
    scaler = TemperatureScaler()
    logits = torch.tensor([-5.0, -3.0, 3.0, 6.0])
    labels = torch.tensor([0, 0, 1, 1])
    scaler.fit(logits, labels)
    assert scaler.temperature.item() > 0.0

    calibrated_probs = scaler.calibrate_probabilities(logits)
    assert calibrated_probs.shape == (4,)
    assert (calibrated_probs >= 0.0).all() and (calibrated_probs <= 1.0).all()


def test_temperature_scaling_monotonicity():
    """Verify temperature scaling preserves relative rank ordering of logits."""
    scaler = TemperatureScaler(initial_temperature=1.5)
    logits = torch.tensor([-2.0, 0.0, 1.5, 4.0])
    probs = scaler.calibrate_probabilities(logits)
    # Monotonically increasing
    assert probs[0] < probs[1] < probs[2] < probs[3]
