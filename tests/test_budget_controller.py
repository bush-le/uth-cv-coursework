"""Unit tests for the hierarchical 2-tier budget and mask allocation controller."""

import pytest
from src.pipelines.budget_controller import AdaptiveBudgetController


def test_budget_allocation_respects_exploration_floor():
    """Verify that even when a mode has 0 error, it receives at least the exploration floor quota."""
    controller = AdaptiveBudgetController(total_candidate_budget=100, min_mode_ratio=0.15)
    # T2I has 0% error, Inpaint has 90% error
    mode_errors = {"sd_text2img": 0.0, "sd_img2img": 0.2, "sd_inpaint": 0.9}
    quotas = controller.compute_mode_quotas(mode_errors)
    
    assert sum(quotas.values()) == 100
    assert quotas["sd_text2img"] >= 15, "sd_text2img must receive at least 15% exploration floor"
    assert quotas["sd_inpaint"] > quotas["sd_img2img"], "High-error inpainting must receive largest budget share"


def test_budget_allocation_uniform_when_no_errors_provided():
    """Verify default uniform distribution across 3 modes when cold-starting without history."""
    controller = AdaptiveBudgetController(total_candidate_budget=150)
    quotas = controller.compute_mode_quotas({})
    assert sum(quotas.values()) == 150
    assert quotas["sd_text2img"] == 50
    assert quotas["sd_inpaint"] == 50
    assert quotas["sd_img2img"] == 50


def test_inpainting_regional_targeting():
    """Verify that region error feedback steers mask target selection."""
    controller = AdaptiveBudgetController()
    region_errors = {"periocular": 0.85, "lips": 0.10, "nose": 0.05, "full": 0.05}
    targets = controller.compute_inpainting_mask_targets(region_errors, count=20)
    
    assert len(targets) == 20
    periocular_count = targets.count("periocular")
    assert periocular_count >= 8, f"Expected dominant periocular masks, got {periocular_count}"
    assert "lips" in targets or "full" in targets, "Exploration floor must retain non-zero secondary targets"


def test_demographic_shield_enforces_balance():
    """Verify demographic shield prevents overconcentration into a single demographic group."""
    controller = AdaptiveBudgetController()
    raw_candidates = [
        {"image_id": f"c_{i}", "demographics": {"gender": "male", "age_group": "young"}}
        for i in range(80)
    ] + [
        {"image_id": f"c_{i+80}", "demographics": {"gender": "female", "age_group": "elderly"}}
        for i in range(20)
    ]
    
    shielded = controller.enforce_demographic_shield(raw_candidates, max_select=40)
    assert len(shielded) <= 40
    genders = [c["demographics"]["gender"] for c in shielded]
    female_ratio = genders.count("female") / len(genders)
    assert female_ratio >= 0.25, f"Demographic shield must protect minority groups, got ratio {female_ratio}"
