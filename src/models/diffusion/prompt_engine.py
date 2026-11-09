"""Dynamic Combinatorial Grammar Engine with Softmax Error-Steering for SD Deepfake Prompts.

Produces rich, photorealistic, highly diverse facial generation prompts across
multiple demographic, optical, lighting, and occlusion slots, while adaptively
steering generation toward vulnerability patterns uncovered by downstream detectors.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.common.logger import get_logger
from src.data.provenance import sample_seed

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Broad Combinatorial Taxonomy
# -----------------------------------------------------------------------------

SLOT_TAXONOMY: Dict[str, List[str]] = {
    "subject_age": [
        "young adult portrait, fresh skin, vibrant natural complexion",
        "middle-aged person portrait, subtle laugh lines, mature refined features",
        "elderly individual face, natural skin wrinkles, graceful aging, silver hair strands",
        "college-age youth portrait, smooth texture, clear natural eyes",
        "mature adult face, defined jawline, authentic facial contours",
        "senior person portrait, dignified expression, soft age marks",
    ],
    "subject_gender": [
        "man face, natural masculine facial contours",
        "woman face, authentic feminine features, natural skin",
        "person portrait, neutral authentic face",
    ],
    "optics": [
        "shot on 85mm f/1.4 prime lens, creamy background bokeh, ultra sharp eye focus",
        "shot on 50mm portrait prime, balanced facial proportions, crisp eye details",
        "shot on 35mm film camera, subtle analog film grain, authentic depth",
        "medium format Hasselblad portrait, natural skin pores, realistic micro-contrast",
        "shot on 85mm portrait prime lens, balanced facial proportions, sharp iris reflections",
        "classic vintage 85mm portrait lens, warm organic film aesthetic, sharp facial details",
    ],
    "lighting": [
        "soft studio key light, diffuse beauty dish fill, subtle catchlights in eyes",
        "direct harsh camera flash, balanced studio fill, authentic exposure",
        "warm golden hour sunlight, low sun angle, radiant natural backlight",
        "diffuse north-facing window light, soft gradient shadows, gentle ambient illumination",
        "dramatic cinematic side lighting, chiascurro shadows, high-contrast cheekbones",
        "cool overcast daylight, even exposure, true-to-life color rendering",
        "subtle colorful rim lighting, studio beauty glow, sculpted facial contours",
        "cozy candlelit warm glow, intimate low-key illumination, deep soft shadows",
    ],
    "expression": [
        "subtle warm smile, natural laughing eyes, relaxed facial muscles",
        "candid neutral glance, contemplative look, natural resting expression",
        "slight confident smirk, focused gaze toward camera, authentic presence",
        "candid sideways glance, dynamic subtle head tilt, spontaneous capture",
        "calm serene expression, soft composed lips, direct honest eye contact",
    ],
    "occlusion": [
        "wearing thin gold wireframe eyeglasses with subtle lens reflections",
        "wearing classic tortoiseshell prescription glasses, distinct stylish frames",
        "wearing dark designer sunglasses with glossy reflection of environment",
        "stray strands of hair falling gently across forehead, candid natural hair",
        "high knit sweater collar framing the chin and jawline",
        "minimalist silver stud earring with tiny specular highlight",
        "bare face with unobstructed natural hairline and clear temples",
    ],
    "imperfections": [
        "authentic skin texture, visible fine pores, microscopic skin imperfections",
        "subtle constellation of natural freckles across the nose bridge and cheeks",
        "delicate laugh lines around mouth corners, authentic organic expressions",
        "subtle natural asymmetrical eyebrows, realistic unedited human face",
        "natural skin pores, fine epidermal texture, subtle organic micro-features",
    ],
}


class PromptGrammarEngine:
    """Dynamic combinatorial grammar engine for photorealistic SD facial prompt synthesis.
    
    Generates deterministic, highly diverse prompts across 7 combinatorial slots
    and adaptively shifts slot selection probabilities toward detector failure patterns.
    """

    def __init__(
        self,
        seed: int = 42,
        exploration_floor: float = 0.15,
    ) -> None:
        """Initialize the prompt grammar engine.

        Args:
            seed: Master random seed for deterministic generation.
            exploration_floor: Minimum probability floor allocated uniformly across all tags
                               in each slot to guarantee non-zero exploration diversity.
        """
        self.seed = seed
        self.exploration_floor = max(0.01, min(0.50, float(exploration_floor)))
        self.taxonomy = {slot: list(tags) for slot, tags in SLOT_TAXONOMY.items()}
        
        # Initialize uniform probability weights for all slots
        self.slot_weights: Dict[str, Dict[str, float]] = {}
        for slot, tags in self.taxonomy.items():
            self.slot_weights[slot] = {tag: 1.0 for tag in tags}

    def update_error_weights(
        self,
        error_stats: Dict[str, float],
        temperature: float = 1.0,
    ) -> None:
        """Update slot selection weights based on detector False Negative / error attribution.

        Args:
            error_stats: Dictionary mapping 'slot:tag' or tag substrings to observed error rates in [0.0, 1.0].
            temperature: Softmax scaling temperature. Lower values (< 1.0) make error steering sharper.
        """
        temp = max(0.1, float(temperature))

        for slot, tags in self.taxonomy.items():
            raw_scores = []
            for tag in tags:
                # Find matching error rate: exact match, 'slot:tag', or substring match
                key_full = f"{slot}:{tag}"
                matched_err = error_stats.get(key_full)
                if matched_err is None:
                    matched_err = error_stats.get(tag)
                if matched_err is None:
                    # Check partial matches
                    matches = [
                        err
                        for err_key, err in error_stats.items()
                        if (":" in err_key and err_key.split(":", 1)[1] in tag) or err_key in tag
                    ]
                    matched_err = max(matches) if matches else 0.0

                raw_scores.append(matched_err)

            scores_arr = np.array(raw_scores, dtype=np.float64) / temp
            # Stable softmax
            exp_scores = np.exp(scores_arr - np.max(scores_arr))
            softmax_probs = exp_scores / np.sum(exp_scores)

            # Blend with uniform exploration floor to guarantee continuous broad coverage
            uniform_prob = 1.0 / len(tags)
            floor = self.exploration_floor
            blended_probs = (1.0 - floor) * softmax_probs + floor * uniform_prob
            blended_probs = blended_probs / np.sum(blended_probs)

            # Store updated normalized probabilities
            for tag, prob in zip(tags, blended_probs):
                self.slot_weights[slot][tag] = float(prob)

        logger.info("Updated PromptGrammarEngine error weights across %d slots.", len(self.taxonomy))

    def get_slot_probabilities(self) -> Dict[str, Dict[str, float]]:
        """Return the current normalized probability distribution for all slots and tags."""
        return {slot: dict(weights) for slot, weights in self.slot_weights.items()}

    def _sample_slot_tag(self, slot: str, rng: np.random.RandomState) -> str:
        """Sample one tag from a slot according to its current probability distribution."""
        tags = self.taxonomy[slot]
        weights = [self.slot_weights[slot][t] for t in tags]
        probs = np.array(weights, dtype=np.float64)
        probs = probs / np.sum(probs)
        chosen_idx = rng.choice(len(tags), p=probs)
        return tags[chosen_idx]

    def generate_prompt(
        self,
        demographics: Optional[Dict[str, Any]] = None,
        mode: str = "text2img",
        seed_offset: int = 0,
    ) -> str:
        """Synthesize a complete, photorealistic prompt incorporating diversity & detector steering.

        Args:
            demographics: Optional demographic constraints from reference face (age_group, gender, glasses).
            mode: Generation mode ('text2img', 'inpaint', 'img2img').
            seed_offset: Per-sample offset to combine with master seed for strict reproducibility.

        Returns:
            Coherent, high-diversity prompt string.
        """
        sample_s = sample_seed(self.seed, f"prompt_{mode}", seed_offset)
        rng = np.random.RandomState(sample_s)

        # 1. Base photographic quality anchor (head-and-shoulders framing prevents both extreme crops and distant mirror scenes)
        if mode == "text2img":
            base_prefix = "head and shoulders portrait, centered, head and hair fully in frame, direct eye contact"
        else:
            base_prefix = "head and shoulders portrait, centered face, authentic direct eye contact"

        # 2. Subject demographics (strictly respect demographic dict if provided)
        age_tag = ""
        gender_tag = ""
        occlusion_tag = ""

        if demographics:
            age = demographics.get("age_group")
            if age in ["0-2", "3-6"]:
                age_tag = "toddler portrait, soft delicate natural skin, gentle baby features"
            elif age in ["7-9", "10-14", "young"]:
                age_tag = "young person portrait, soft natural skin, youthful facial features"
            elif age in ["50-69", "70-120", "mature"]:
                age_tag = "elderly individual face, natural skin wrinkles, dignified aging, silver hair"
            else:
                age_tag = "adult portrait, authentic facial contours, natural skin"

            gender = demographics.get("gender")
            if gender == "male":
                gender_tag = "man face, natural masculine facial contours"
            elif gender == "female":
                gender_tag = "woman face, authentic feminine features, natural skin"
            else:
                gender_tag = self._sample_slot_tag("subject_gender", rng)

            glasses = demographics.get("glasses")
            if glasses == "Dark":
                occlusion_tag = "wearing dark designer sunglasses with subtle lens glare"
            elif glasses in ("Normal", "Glasses") or demographics.get("eyeglasses") is True:
                occlusion_tag = "wearing eyeglasses with realistic frame reflections"
            elif mode != "inpaint":
                occlusion_tag = self._sample_slot_tag("occlusion", rng)
            else:
                occlusion_tag = ""
        else:
            age_tag = self._sample_slot_tag("subject_age", rng)
            gender_tag = self._sample_slot_tag("subject_gender", rng)
            occlusion_tag = self._sample_slot_tag("occlusion", rng) if mode != "inpaint" else ""

        # 3. Photographic slots: Optics, Lighting, Expression, Micro-imperfections
        optics_tag = self._sample_slot_tag("optics", rng)
        lighting_tag = self._sample_slot_tag("lighting", rng)
        expression_tag = self._sample_slot_tag("expression", rng)
        imperfection_tag = self._sample_slot_tag("imperfections", rng)

        # Assemble prompt based on mode specialization
        if mode == "inpaint":
            # For inpainting, focus heavily on seamless local blends and facial detail
            prompt_parts = [
                "seamless natural facial features, authentic micro skin texture",
                gender_tag,
                age_tag,
                occlusion_tag,
                expression_tag,
                lighting_tag,
            ]
        elif mode == "img2img":
            # For img2img, focus on style preservation and realistic skin illumination
            prompt_parts = [
                "high resolution authentic portrait, natural skin texture, realistic illumination",
                gender_tag,
                age_tag,
                occlusion_tag,
                optics_tag,
                lighting_tag,
                imperfection_tag,
            ]
        else:
            # Full text2image: comprehensive global and local synthesis
            prompt_parts = [
                base_prefix,
                gender_tag,
                age_tag,
                occlusion_tag,
                lighting_tag,
                optics_tag,
                expression_tag,
                imperfection_tag,
            ]

        # Filter empty strings and assemble full phrases without mid-sentence clipping
        cleaned_parts = [p.strip() for p in prompt_parts if p.strip()]
        selected_parts = []
        cur_words = 0
        for part in cleaned_parts:
            part_words = len(part.split())
            if cur_words + part_words <= 64:
                selected_parts.append(part)
                cur_words += part_words
            else:
                break
        return ", ".join(selected_parts)
