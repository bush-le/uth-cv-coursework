"""Unit tests for the dynamic combinatorial prompt grammar engine."""

import pytest
from src.models.diffusion.prompt_engine import PromptGrammarEngine


def test_prompt_grammar_deterministic_generation():
    """Verify identical seeds produce identical prompts across calls."""
    engine = PromptGrammarEngine(seed=42)
    p1 = engine.generate_prompt(mode="text2img", seed_offset=100)
    p2 = engine.generate_prompt(mode="text2img", seed_offset=100)
    assert p1 == p2
    assert isinstance(p1, str)
    assert len(p1) > 20


def test_prompt_grammar_rich_taxonomy_coverage():
    """Verify the taxonomy generates broad diverse demographic, lighting, and camera styles."""
    engine = PromptGrammarEngine(seed=42)
    prompts = [engine.generate_prompt(mode="text2img", seed_offset=i) for i in range(100)]
    
    # Check that multiple diverse descriptors appear across samples
    unique_prompts = set(prompts)
    assert len(unique_prompts) >= 95, "Prompts should be highly diverse (>= 95 unique per 100)!"

    # Verify presence of multiple photographic optics
    has_lenses = any("mm" in p for p in prompts)
    assert has_lenses, "Taxonomy must include camera lens optics."

    # Verify presence of natural lighting descriptors
    lighting_keywords = ["lighting", "flash", "sunlight", "golden hour", "ambient", "shadow", "overcast", "cinematic"]
    has_lighting = any(any(k in p for k in lighting_keywords) for p in prompts)
    assert has_lighting, "Taxonomy must include diverse lighting conditions."


def test_prompt_grammar_demographic_context_integration():
    """Verify demographic overrides take precedence while enriching surrounding descriptors."""
    engine = PromptGrammarEngine(seed=42)
    demo = {
        "age_group": "50-69",
        "gender": "female",
        "glasses": "Dark",
    }
    prompt = engine.generate_prompt(demographics=demo, mode="inpaint", seed_offset=10)
    assert "woman" in prompt or "female" in prompt or "elderly" in prompt or "mature" in prompt
    assert "sunglasses" in prompt or "glasses" in prompt


def test_prompt_grammar_error_weight_adaptation():
    """Verify that error statistics dynamically steer slot selection probabilities."""
    engine = PromptGrammarEngine(seed=42, exploration_floor=0.15)
    
    # Error stats indicate high failure on direct flash lighting
    error_stats = {
        "lighting:direct harsh camera flash": 0.95,
        "lighting:soft studio key light": 0.05,
    }
    engine.update_error_weights(error_stats, temperature=0.5)
    probs = engine.get_slot_probabilities()

    flash_key = next(k for k in probs["lighting"] if "direct harsh camera flash" in k)
    studio_key = next(k for k in probs["lighting"] if "soft studio key light" in k)

    assert probs["lighting"][flash_key] > probs["lighting"][studio_key]
    
    # Check generated batch reflects error steering
    steered_prompts = [engine.generate_prompt(mode="text2img", seed_offset=i) for i in range(50)]
    flash_count = sum(1 for p in steered_prompts if "camera flash" in p)
    assert flash_count >= 10, f"Expected elevated direct camera flash occurrences, got {flash_count}"
