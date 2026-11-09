"""Hierarchical 2-Tier Modality Budget & Regional Mask Allocation Controller.

Dynamically allocates candidate generation budgets across synthesis modalities
(T2I, I2I, Inpainting) and guides regional inpainting masks based on downstream
detector error feedback, while strictly shielding demographic diversity.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.common.logger import get_logger

logger = get_logger(__name__)

STANDARD_MODES = ["sd_text2img", "sd_inpaint", "sd_img2img"]
STANDARD_REGIONS = ["periocular", "lips", "nose", "full"]


class AdaptiveBudgetController:
    """Orchestrates 2-Tier budget and mask allocation for closed-loop active learning.
    
    Tier 1: Allocates generation quotas among T2I, I2I, and Inpainting based on mode-level
            False Negative rates, enforcing a strict minimum exploration floor.
    Tier 2: Directs Inpainting masks toward vulnerable facial regions (eyes, mouth, nose, full)
            and applies a Demographic Shield to maintain balanced demographic representation.
    """

    def __init__(
        self,
        total_candidate_budget: int = 150,
        min_mode_ratio: float = 0.15,
        exploration_floor: float = 0.20,
        seed: int = 42,
    ) -> None:
        """Initialize the budget controller.

        Args:
            total_candidate_budget: Total number of candidate images to synthesize in this round.
            min_mode_ratio: Hard lower bound fraction guaranteed to every mode (default 15%).
            exploration_floor: Uniform exploration probability blend across candidate options.
            seed: Master seed for deterministic allocation.
        """
        self.total_candidate_budget = max(3, int(total_candidate_budget))
        self.min_mode_ratio = max(0.05, min(0.30, float(min_mode_ratio)))
        self.exploration_floor = max(0.05, min(0.50, float(exploration_floor)))
        self.seed = seed

    def compute_mode_quotas(
        self,
        mode_errors: Optional[Dict[str, float]] = None,
        modes: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Allocate integer candidate quotas across generation modes based on observed detector errors.

        Args:
            mode_errors: Mapping from mode name to error rate in [0.0, 1.0].
            modes: List of active modes (defaults to T2I, Inpaint, I2I).

        Returns:
            Dictionary mapping each mode to its integer candidate allocation, summing to total_candidate_budget.
        """
        active_modes = modes or STANDARD_MODES
        k = len(active_modes)
        if k == 0:
            return {}

        if not mode_errors:
            # Uniform split
            base_count = self.total_candidate_budget // k
            remainder = self.total_candidate_budget % k
            quotas = {m: base_count for m in active_modes}
            for i in range(remainder):
                quotas[active_modes[i]] += 1
            return quotas

        # Extract error rates with fallbacks
        raw_errs = [max(0.0, float(mode_errors.get(m, 0.0))) for m in active_modes]
        err_arr = np.array(raw_errs, dtype=np.float64)

        if np.all(err_arr == 0.0) or np.all(err_arr == err_arr[0]):
            softmax_probs = np.ones(k, dtype=np.float64) / k
        else:
            exp_err = np.exp(err_arr - np.max(err_arr))
            softmax_probs = exp_err / np.sum(exp_err)

        # Enforce minimum mode exploration ratio
        min_p = self.min_mode_ratio
        clamped_probs = np.maximum(softmax_probs, min_p)
        final_probs = clamped_probs / np.sum(clamped_probs)

        # Largest Remainder Method (Hare-Niemeyer) for exact integer partitioning
        exact_counts = final_probs * self.total_candidate_budget
        integer_counts = np.floor(exact_counts).astype(int)
        remainder = self.total_candidate_budget - int(np.sum(integer_counts))

        fractional_parts = exact_counts - integer_counts
        sorted_indices = np.argsort(-fractional_parts)
        for idx in sorted_indices[:remainder]:
            integer_counts[idx] += 1

        result = {m: int(integer_counts[i]) for i, m in enumerate(active_modes)}
        logger.info(
            "Adaptive mode quotas allocated (total=%d): %s",
            self.total_candidate_budget,
            result,
        )
        return result

    def compute_inpainting_mask_targets(
        self,
        region_errors: Optional[Dict[str, float]] = None,
        count: int = 50,
        regions: Optional[List[str]] = None,
    ) -> List[str]:
        """Compute ordered list of regional inpainting mask targets guided by regional failure rates.

        Args:
            region_errors: Mapping from facial region ('periocular', 'lips', 'nose', 'full') to error rate.
            count: Number of inpainting mask targets to generate.
            regions: List of available region names.

        Returns:
            List of region target strings of length count.
        """
        active_regions = regions or STANDARD_REGIONS
        r_len = len(active_regions)
        if count <= 0 or r_len == 0:
            return []

        rng = np.random.RandomState(self.seed)

        if not region_errors:
            # Uniform cyclic targets
            return [active_regions[i % r_len] for i in range(count)]

        raw_errs = [max(0.0, float(region_errors.get(r, 0.0))) for r in active_regions]
        err_arr = np.array(raw_errs, dtype=np.float64)

        if np.all(err_arr == 0.0) or np.all(err_arr == err_arr[0]):
            probs = np.ones(r_len, dtype=np.float64) / r_len
        else:
            exp_err = np.exp(err_arr - np.max(err_arr))
            softmax_probs = exp_err / np.sum(exp_err)
            # Blend with 20% exploration floor
            uniform_p = 1.0 / r_len
            probs = (1.0 - self.exploration_floor) * softmax_probs + self.exploration_floor * uniform_p
            probs = probs / np.sum(probs)

        sampled_indices = rng.choice(r_len, size=count, p=probs)
        return [active_regions[idx] for idx in sampled_indices]

    def enforce_demographic_shield(
        self,
        candidate_records: List[Dict[str, Any]],
        max_select: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Stratify candidate samples across demographic groups to prevent shortcut learning collapse.

        Groups records by demographic tuple (gender, age_group) and draws round-robin samples
        so that no single group monopolizes the candidate pool.

        Args:
            candidate_records: List of candidate metadata dictionaries.
            max_select: Maximum number of records to return. Defaults to len(candidate_records).

        Returns:
            Balanced list of candidate metadata dictionaries.
        """
        if not candidate_records:
            return []

        limit = max_select if max_select is not None else len(candidate_records)
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for rec in candidate_records:
            demo = rec.get("demographics") or {}
            gender = demo.get("gender", "unknown")
            age = demo.get("age_group", "unknown")
            key = f"{gender}_{age}"
            groups.setdefault(key, []).append(rec)

        group_keys = sorted(groups.keys())
        # Shuffle each bucket deterministically
        rng = np.random.RandomState(self.seed)
        for k in group_keys:
            perm = rng.permutation(len(groups[k]))
            groups[k] = [groups[k][i] for i in perm]

        selected: List[Dict[str, Any]] = []
        idx = 0
        while len(selected) < limit:
            added_in_round = False
            for k in group_keys:
                if idx < len(groups[k]):
                    selected.append(groups[k][idx])
                    added_in_round = True
                    if len(selected) >= limit:
                        break
            if not added_in_round:
                break
            idx += 1

        logger.info(
            "Demographic shield applied: retained %d/%d candidates across %d demographic groups.",
            len(selected),
            len(candidate_records),
            len(group_keys),
        )
        return selected
