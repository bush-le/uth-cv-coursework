from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
DECK_PATH = REPO_ROOT / "docs/reports/presentation_5min_define_and_design.html"


def _extract_visible_text(html: str) -> str:
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", no_style, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", no_script)
    return re.sub(r"\s+", " ", text).strip()


def test_deck_has_six_slides_and_fixed_stage():
    html = DECK_PATH.read_text(encoding="utf-8")
    assert "width: 1920px;" in html and "height: 1080px;" in html
    slides = re.findall(r'<section[^>]+class="[^"]*slide[^"]*"[^>]*>', html)
    assert len(slides) == 6
    for idx in range(1, 7):
        assert f'id="slide-{idx}"' in html


def test_deck_enforces_blue_professional_single_accent_palette():
    html = DECK_PATH.read_text(encoding="utf-8").lower()
    assert "#fdfae7" in html
    assert "#1e2bfa" in html
    for forbidden_hex in ["#d97706", "#dc2626", "#059669", "#9333ea", "#1e293b", "#111827"]:
        assert forbidden_hex not in html, f"Forbidden non-Blue-Professional color {forbidden_hex}"
    assert "notes-drawer" not in html
    assert "<pre" not in html


def test_deck_contains_zero_unverified_numbers_or_code_clutter():
    html = DECK_PATH.read_text(encoding="utf-8")
    visible = _extract_visible_text(html)
    forbidden_patterns = [
        r"\b95%\b",
        r"\b62%\b",
        r"\b92%\b",
        r"\b85%\b",
        r"\b50\.0\b",
        r"&ge;\s*50",
        r"≥\s*50",
        r"\b0\.28\b",
        r"\b0\.30\b",
        r"\b0\.85\b",
        r"\b78\.4\b",
        r"\b0\.312\b",
        r"\b3,000\b",
        r"\b5,000\b",
        r"\b256\b",
        r"\b512\b",
        r"\b70%\b",
        r"\b15%\b",
        r"\b6GB\b",
        r"\b50 steps\b",
        r"\b30 steps\b",
        r"\bSHA-256\b",
        r"\bFP16\b",
        r"\b1e-4\b",
        r"\bsrc/",
        r"\bdata/",
    ]
    for pat in forbidden_patterns:
        assert not re.search(pat, visible, flags=re.I), f"Forbidden unverified detail matched: {pat}"


def test_every_slide_has_inline_svg_visual_diagram():
    html = DECK_PATH.read_text(encoding="utf-8")
    parts = re.split(r'<section[^>]+class="[^"]*slide[^"]*"[^>]*>', html)[1:]
    assert len(parts) == 6
    for idx, slide_html in enumerate(parts, start=1):
        assert "<svg" in slide_html, f"Slide {idx} is missing an inline <svg> visual diagram"
