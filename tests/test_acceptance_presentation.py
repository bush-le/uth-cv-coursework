"""Unit tests for the Final Acceptance Scientific Report (Vertical Format).

Verifies the integrity, section structure, interactive navigation, and visual artifact
references of docs/reports/presentation_final_acceptance.html.
"""

from pathlib import Path
import pytest
from bs4 import BeautifulSoup

PRESENTATION_PATH = Path("docs/reports/presentation_final_acceptance.html")


@pytest.mark.smoke
def test_acceptance_presentation_exists_and_size():
    """Verify presentation file exists and is within expected size bounds."""
    assert PRESENTATION_PATH.is_file(), f"Missing file: {PRESENTATION_PATH}"
    size_kb = PRESENTATION_PATH.stat().st_size / 1024
    assert 20 < size_kb < 500, f"Unexpected size: {size_kb:.1f} KB"


@pytest.mark.smoke
def test_acceptance_presentation_vertical_structure():
    """Verify DOM structure: layout-wrapper, sidebar-nav, report-sections, and top-header."""
    content = PRESENTATION_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    # Check top header and sidebar
    header = soup.find("header", class_="top-header")
    assert header is not None, "Missing top-header"

    sidebar = soup.find("nav", class_="sidebar-nav")
    assert sidebar is not None, "Missing sidebar-nav"
    nav_links = sidebar.find_all("a", class_="nav-link")
    assert len(nav_links) >= 12, f"Expected at least 12 nav links including 03b, found {len(nav_links)}"

    # Check report sections
    sections = soup.find_all("section", class_="report-section")
    assert len(sections) >= 12, f"Expected at least 12 report sections including 03b, found {len(sections)}"
    gen_sec = soup.find("section", id="sec-generation-mechanics")
    assert gen_sec is not None, "Missing #sec-generation-mechanics (03b) section"

    # Check KPI cards
    kpis = soup.find_all("div", class_="kpi-card")
    assert len(kpis) >= 4, f"Expected at least 4 KPI cards, found {len(kpis)}"


@pytest.mark.smoke
def test_acceptance_presentation_visual_artifacts_exist_on_disk():
    """Verify that 100% of embedded visual artifact images exist on disk with valid file size."""
    content = PRESENTATION_PATH.read_text(encoding="utf-8")
    soup = BeautifulSoup(content, "html.parser")

    # Filter out empty template image in lightbox modal
    imgs = [img for img in soup.find_all("img") if img.get("id") != "lightbox-img"]
    assert len(imgs) >= 15, f"Expected at least 15 real visual artifacts and face samples, found {len(imgs)}"

    reports_dir = PRESENTATION_PATH.parent
    for img in imgs:
        src = img.get("src")
        assert src, "Image tag missing src attribute"
        target_path = (reports_dir / src).resolve()
        assert target_path.is_file(), f"Broken image artifact link: {src} -> {target_path}"
        assert target_path.stat().st_size > 1000, f"Artifact file is unexpectedly empty: {target_path}"


@pytest.mark.smoke
def test_acceptance_presentation_scientific_metrics():
    """Verify that core empirical facts and champion metrics are correctly stated."""
    content = PRESENTATION_PATH.read_text(encoding="utf-8")

    # Check key figures and terms
    assert "6.000" in content, "Must mention 6,000 synthetic images"
    assert "213" in content, "Must mention 213 Challenge Vault samples"
    assert "93.13%" in content, "Must mention Champion M3 Quad-Vault recall 93.13%"
    assert "91.89%" in content, "Must mention M2* Challenge Vault recall 91.89%"
    assert "11.71%" in content, "Must mention M0 Challenge Vault recall 11.71%"
    assert "20.400" in content, "Must mention 20,400 Universal Benchmark images"
    assert "Zero Identity Leakage" in content or "Zero Leakage" in content
    assert "Soft-Mask Feathering" in content or "Soft Feathering" in content
    assert "CurationGate" in content
    assert "DINOv2" in content
    for term in (
        "PromptGrammarEngine",
        "DPMSolverMultistepScheduler",
        "CALIBRATED_NEGATIVE_PROMPT",
        "2.000",
        "blur_radius",
    ):
        assert term in content, f"Missing generation mechanics technical detail: {term}"

    # Verify visual flow & absence of debunked factual errors
    assert "pipeline-flow-grid" in content, "Missing interactive visual pipeline flow map"
    assert "Dlib 68" not in content, "Outdated Dlib 68 claim must not appear (uses opencv_haar)"
    assert "Midjourney" not in content, "Midjourney is not in the 34 universal benchmark methods"
    assert "Stable Diffusion XL" not in content, "SDXL is not in the 34 universal benchmark methods"
    assert "benchmark_4models_global_comparison.png" not in content, "Broken 4-model global chart must not be embedded"
