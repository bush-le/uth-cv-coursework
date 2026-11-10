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


def test_infant_prompt_does_not_contain_adult_gender_words() -> None:
    """Verify toddler/infant demographics never inject adult 'man'/'woman' chimera words."""
    engine = PromptGrammarEngine(seed=42)
    demo = {"age_group": "0-2", "gender": "male"}
    prompt = engine.generate_prompt(demographics=demo, mode="text2img", seed_offset=1)
    prompt_words = prompt.lower().replace(",", " ").split()
    assert "man" not in prompt_words, f"Adult word 'man' found in toddler prompt: {prompt}"
    assert "baby boy" in prompt.lower() or "toddler" in prompt.lower()


def test_young_adult_demographics_no_toddler_collision() -> None:
    """Verify '20-29' age group does not falsely collide with '0-2' toddler substrings."""
    engine = PromptGrammarEngine(seed=42)
    demo_20_29 = {"age_group": "20-29", "gender": "female"}
    prompt = engine.generate_prompt(demographics=demo_20_29, mode="inpaint", seed_offset=4)
    assert "baby" not in prompt.lower()
    assert "toddler" not in prompt.lower()
    assert "woman face" in prompt.lower() or "adult" in prompt.lower()


def test_prompt_word_count_within_clip_budget() -> None:
    """Verify all generated prompts across 50 offsets stay <= 45 words (< 70 CLIP tokens)."""
    engine = PromptGrammarEngine(seed=42)
    for mode in ["text2img", "inpaint", "img2img"]:
        for i in range(20):
            p = engine.generate_prompt(mode=mode, seed_offset=i)
            word_count = len(p.split())
            assert word_count <= 45, f"Mode {mode} offset {i} exceeded 45 words ({word_count}): {p}"


def test_error_steering_nan_inf_immunity() -> None:
    """Verify that update_error_weights immunizes against NaN and Inf error metrics without poisoning."""
    import numpy as np

    engine = PromptGrammarEngine(seed=42)
    corrupted_stats = {
        "lighting:direct harsh camera flash": float("nan"),
        "lighting:soft studio key light": float("inf"),
        "optics:hasselblad": -0.5,
        "expression:smile": 1.5,
    }
    engine.update_error_weights(corrupted_stats, temperature=1.0)
    probs = engine.get_slot_probabilities()
    for slot, slot_probs in probs.items():
        for tag, prob in slot_probs.items():
            assert not np.isnan(prob), f"Found NaN prob in slot {slot}, tag {tag}"
            assert not np.isinf(prob), f"Found Inf prob in slot {slot}, tag {tag}"
            assert 0.0 <= prob <= 1.0

    # Ensure sampling doesn't crash
    p = engine.generate_prompt(mode="text2img", seed_offset=1)
    assert isinstance(p, str) and len(p) > 10


def test_error_steering_temperature_boundary() -> None:
    """Verify extreme temperatures (0.0, negative, nan, inf) do not crash or produce NaNs."""
    import numpy as np

    engine = PromptGrammarEngine(seed=42)
    stats = {"lighting:direct harsh camera flash": 0.8}
    for t in [0.0, -2.0, float("nan"), float("inf"), 1e6]:
        engine.update_error_weights(stats, temperature=t)
        probs = engine.get_slot_probabilities()
        for slot, slot_probs in probs.items():
            for tag, prob in slot_probs.items():
                assert not np.isnan(prob)
                assert not np.isinf(prob)


def test_demographic_glasses_none_no_glasses_injection() -> None:
    """Verify that when demographics explicitly state glasses='None', no glasses tags are injected."""
    engine = PromptGrammarEngine(seed=42)
    demographics = {"dataset": "ffhq", "age_group": "20-29", "gender": "male", "glasses": "None"}
    for mode in ["text2img", "img2img", "inpaint"]:
        for offset in range(20):
            p = engine.generate_prompt(demographics=demographics, mode=mode, seed_offset=offset)
            assert "glasses" not in p.lower(), f"Unexpected glasses in prompt for glasses=None: {p}"
            assert "sunglasses" not in p.lower(), f"Unexpected sunglasses in prompt for glasses=None: {p}"



