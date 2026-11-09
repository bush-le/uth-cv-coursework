"""Temperature Calibration and Calibrated Dual-Criterion Hard Sample Mining.

Extracts high-value adversarial hard samples (False Negatives + Boundary Uncertain)
guided by temperature-scaled detector confidence, and filters out perceptual duplicates
to maximize dataset utility without redundant clustering.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS

from src.common.logger import get_logger
from src.data.image_schema import ImageMetadata

logger = get_logger(__name__)


class CalibratedHardMiner:
    """Mines hard synthetic deepfakes using calibrated detector confidence and diversity filtering."""

    def __init__(
        self,
        temperature: float = 1.0,
        decision_threshold: float = 0.5,
        uncertainty_window: float = 0.10,
        min_cosine_distance: float = 0.05,
    ) -> None:
        """Initialize the miner.

        Args:
            temperature: Softmax/logit temperature scale parameter T > 0.
            decision_threshold: Classification decision threshold tau (default 0.5).
            uncertainty_window: Boundary delta around tau where samples are considered uncertain.
            min_cosine_distance: Minimum cosine distance (1 - cos_sim) required between hard samples.
        """
        self.temperature = max(0.01, float(temperature))
        self.decision_threshold = float(decision_threshold)
        self.uncertainty_window = float(uncertainty_window)
        self.min_cosine_distance = float(min_cosine_distance)

    def calibrate_temperature(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """Fit scalar temperature T on a held-out validation set via NLL minimization.

        Args:
            val_logits: Unscaled logit tensor of shape (N,) or (N, 1).
            val_labels: Target binary label tensor of shape (N,) or (N, 1).
            max_iter: Maximum optimization iterations.

        Returns:
            Optimized temperature float.
        """
        logits = val_logits.view(-1).float()
        labels = val_labels.view(-1).float()
        if len(logits) == 0:
            return self.temperature

        # Parameterize temperature as exp(log_temp) to guarantee T > 0
        log_temp = nn.Parameter(torch.tensor([np.log(self.temperature)], dtype=torch.float32))
        criterion = nn.BCEWithLogitsLoss()
        optimizer = LBFGS([log_temp], lr=0.05, max_iter=max_iter)

        def eval_step() -> torch.Tensor:
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            loss = criterion(logits / temp, labels)
            loss.backward()
            return loss

        try:
            optimizer.step(eval_step)
            fitted_temp = float(torch.exp(log_temp).item())
            self.temperature = max(0.01, min(10.0, fitted_temp))
            logger.info("Fitted validation temperature T = %.4f", self.temperature)
        except Exception as e:
            logger.warning("Temperature calibration failed (%s); retaining T = %.4f", e, self.temperature)

        return self.temperature

    def apply_temperature_to_probabilities(self, probs: List[float]) -> List[float]:
        """Scale probabilities using current temperature T."""
        if abs(self.temperature - 1.0) < 1e-4:
            return probs

        calibrated = []
        for p in probs:
            p_clamped = max(1e-6, min(1.0 - 1e-6, float(p)))
            logit = np.log(p_clamped / (1.0 - p_clamped))
            cal_logit = logit / self.temperature
            cal_p = 1.0 / (1.0 + np.exp(-cal_logit))
            calibrated.append(float(cal_p))
        return calibrated

    def mine_hard_samples(
        self,
        candidate_records: List[ImageMetadata],
        probabilities: List[float],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[ImageMetadata], Dict[str, Any]]:
        """Mine adversarial hard samples based on False Negative and Boundary Uncertainty criteria.

        Args:
            candidate_records: Candidate metadata objects.
            probabilities: Raw or calibrated P(fake) probabilities from detector.
            embeddings: Optional (N, D) feature embeddings for perceptual diversity deduplication.

        Returns:
            Tuple of (mined_hard_samples, mining_statistics).
        """
        if len(candidate_records) != len(probabilities):
            raise ValueError(
                f"Length mismatch: {len(candidate_records)} candidates vs {len(probabilities)} probabilities."
            )

        cal_probs = self.apply_temperature_to_probabilities(probabilities)

        fn_candidates: List[Tuple[ImageMetadata, float, int]] = []
        unc_candidates: List[Tuple[ImageMetadata, float, int]] = []

        tau = self.decision_threshold
        delta = self.uncertainty_window

        for idx, (meta, p) in enumerate(zip(candidate_records, cal_probs)):
            if meta.label != 1:
                # Only fake images can be adversarial hard deepfakes
                continue

            is_fn = p < tau
            is_uncertain = abs(p - tau) <= delta

            if is_fn:
                fn_candidates.append((meta, p, idx))
            elif is_uncertain:
                unc_candidates.append((meta, p, idx))

        logger.info(
            "Identified %d False Negatives and %d Boundary Uncertain candidates.",
            len(fn_candidates),
            len(unc_candidates),
        )

        # Merge candidate list (prioritizing False Negatives first, then Uncertain)
        combined_candidates = fn_candidates + unc_candidates

        # Apply perceptual diversity filter if embeddings are available
        selected_records: List[ImageMetadata] = []
        selected_embs: List[np.ndarray] = []
        duplicates_filtered = 0

        norm_embeddings = None
        if embeddings is not None and len(embeddings) == len(candidate_records):
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm_embeddings = embeddings / np.maximum(norms, 1e-9)

        for meta, p, idx in combined_candidates:
            if norm_embeddings is not None:
                curr_emb = norm_embeddings[idx]
                is_duplicate = False
                for seen_emb in selected_embs:
                    cosine_sim = float(np.dot(curr_emb, seen_emb))
                    cosine_dist = 1.0 - cosine_sim
                    if cosine_dist < self.min_cosine_distance:
                        is_duplicate = True
                        break

                if is_duplicate:
                    duplicates_filtered += 1
                    continue

                selected_embs.append(curr_emb)

            hard_meta = ImageMetadata(
                image_id=meta.image_id,
                label=1,
                manipulation_type=meta.manipulation_type,
                prompt=meta.prompt,
                source_real_id=meta.source_real_id,
                file_path=meta.file_path,
                bbox=meta.bbox,
                is_hard_sample=True,
                baseline_score=round(p, 4),
                data_source="hard_sample",
                demographics=meta.demographics,
                generation=meta.generation,
                scores={"detector_p_fake": p, "calibrated_score": p},
            )
            selected_records.append(hard_meta)

        stats = {
            "total_evaluated": len(candidate_records),
            "false_negative_count": len(fn_candidates),
            "uncertain_count": len(unc_candidates),
            "perceptual_duplicates_filtered": duplicates_filtered,
            "mined_hard_samples_count": len(selected_records),
            "hard_sample_yield_rate": (
                round(len(selected_records) / max(1, len(candidate_records)), 4)
            ),
            "temperature_used": self.temperature,
        }
        logger.info("Mining complete: %s", stats)
        return selected_records, stats
