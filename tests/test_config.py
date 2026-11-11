"""Unit tests for centralized dataset factory configuration loader and resolver."""

from pathlib import Path
import pytest

from src.common.config import REQUIRED_KEYS, cfg_get, load_config, resolve


def test_cfg_get_missing_key_raises_named_keyerror(tmp_path: Path) -> None:
    """Verify that cfg_get retrieves nested keys and raises named KeyError on missing key."""
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 1\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg_get(cfg, "a.b") == 1
    with pytest.raises(KeyError, match="a.c"):
        cfg_get(cfg, "a.c")


def test_resolve_prefers_cli_over_config(tmp_path: Path) -> None:
    """Verify that resolve() prefers non-None CLI value over configuration."""
    p = tmp_path / "c.yaml"
    p.write_text("x: 3\n", encoding="utf-8")
    cfg = load_config(p)
    assert resolve(None, cfg, "x") == 3
    assert resolve(7, cfg, "x") == 7


def test_repo_factory_config_has_all_required_keys() -> None:
    """Verify that the production configs/dataset_factory.yaml contains all required keys."""
    cfg_path = Path("configs/dataset_factory.yaml")
    assert cfg_path.exists(), "configs/dataset_factory.yaml must exist"
    cfg = load_config(cfg_path)
    for key in REQUIRED_KEYS:
        val = cfg_get(cfg, key)
        assert val is not None or key.endswith("max_real_samples") or key.endswith("detector_checkpoint"), (
            f"Key '{key}' is required and must be defined"
        )
