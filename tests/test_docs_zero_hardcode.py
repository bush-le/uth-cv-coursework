"""Automated verification ensuring documentation contains zero hardcoded model IDs or legacy paths."""

from pathlib import Path
import re
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

DOC_GLOBS = [
    "docs/PURPOSE.md",
    "docs/OVERVIEW.md",
    "README.md",
    "data/README.md",
    "docs/coursework/TAXONOMY_AND_DATA_SPEC.md",
    "docs/phases/PHASE_01_SD_DEEPFAKE_GENERATION_SPEC.md",
    "docs/phases/PHASE_02_DINOVIT_HARD_SAMPLE_MINING_SPEC.md",
    "docs/phases/PHASE_03_CLOSED_LOOP_RETRAINING_SPEC.md",
    "docs/experiments/EXPERIMENT_PROTOCOL.md",
]

FORBIDDEN_PATTERNS = [
    (r"data/processed", "Legacy data/processed path found; use data/synthetic or data/benchmarks"),
    (r"runwayml/", "Hardcoded model ID runwayml/ found; use generation.model_id from config"),
    (r"facebook/dinov2", "Hardcoded model ID facebook/dinov2 found; use hardness.backbone_id from config"),
    (r"5 điểm mốc", "Outdated landmark assumption '5 điểm mốc' found"),
]


def test_docs_contain_no_hardcoded_values_or_legacy_paths():
    """Verify that documentation contains zero hardcoded repository IDs or legacy paths."""
    for rel_path in DOC_GLOBS:
        p = REPO_ROOT / rel_path
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        for pattern, msg in FORBIDDEN_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            assert not match, f"In {rel_path}: {msg} (matched '{match.group(0)}')"


def test_docs_follow_dataset_first_framing():
    """Verify that docs/PURPOSE.md explicitly adopts Dataset-First framing and Proof of Utility."""
    p = REPO_ROOT / "docs/PURPOSE.md"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "dataset" in text.lower()
    assert "Proof of Utility" in text or "proof-of-utility" in text.lower()
