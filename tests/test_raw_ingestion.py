"""Unit tests for real face dataset ingestion (FFHQ and CelebA-HQ)."""

from pathlib import Path
import pytest

from src.data.raw_ingestion import (
    build_baseline_manifest,
    parse_celeba_metadata,
    parse_ffhq_aging_metadata,
)


def test_parse_ffhq_aging_metadata(tmp_path: Path) -> None:
    """Test parsing FFHQ aging CSV."""
    csv_file = tmp_path / "ffhq_test.csv"
    csv_file.write_text(
        "image_number,age_group,gender,glasses,head_pitch,head_roll,head_yaw\n"
        "0,0-2,male,None,4.64,2.17,-9.35\n"
        "1,50-69,female,Dark,8.73,-1.48,6.37\n",
        encoding="utf-8",
    )

    records = parse_ffhq_aging_metadata(csv_file)
    assert len(records) == 2
    assert records[0]["age_group"] == "0-2"
    assert records[0]["gender"] == "male"
    assert records[0]["glasses"] == "None"
    assert records[1]["age_group"] == "50-69"
    assert records[1]["glasses"] == "Dark"


def test_parse_celeba_metadata(tmp_path: Path) -> None:
    """Test parsing CelebA-HQ mapping, identity, and attributes."""
    mapping_file = tmp_path / "mapping.txt"
    mapping_file.write_text(
        "idx orig_idx orig_file\n"
        "0 119613 119614.jpg\n"
        "1 99094 099095.jpg\n",
        encoding="utf-8",
    )

    id_file = tmp_path / "identity.txt"
    id_file.write_text(
        "119614.jpg 7423\n"
        "099095.jpg 7319\n",
        encoding="utf-8",
    )

    attr_file = tmp_path / "attributes.txt"
    attr_file.write_text(
        "2\n"
        "Male Eyeglasses Young\n"
        "0.jpg -1 -1 1\n"
        "1.jpg 1 1 -1\n",
        encoding="utf-8",
    )

    records = parse_celeba_metadata(
        mapping_path=mapping_file,
        identity_path=id_file,
        attribute_path=attr_file,
    )
    assert len(records) == 2
    assert records[0]["identity_id"] == "7423"
    assert records[0]["young"] is True
    assert records[0]["male"] is False
    assert records[1]["identity_id"] == "7319"
    assert records[1]["eyeglasses"] is True


def test_build_baseline_manifest(tmp_path: Path) -> None:
    """Test full baseline manifest building with mock images and metadata."""
    ffhq_dir = tmp_path / "ffhq"
    ffhq_dir.mkdir()
    (ffhq_dir / "00000.png").write_bytes(b"dummy_png_0")
    (ffhq_dir / "00001.png").write_bytes(b"dummy_png_1")

    celeba_dir = tmp_path / "celeba"
    celeba_dir.mkdir()
    (celeba_dir / "0.jpg").write_bytes(b"dummy_jpg_0")
    (celeba_dir / "1.jpg").write_bytes(b"dummy_jpg_1")

    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()
    (meta_dir / "ffhq_aging_labels.csv").write_text(
        "image_number,age_group,gender,glasses\n0,0-2,male,None\n1,30-39,female,Normal\n",
        encoding="utf-8",
    )
    (meta_dir / "CelebA-HQ-to-CelebA-mapping.txt").write_text(
        "idx orig_idx orig_file\n0 100 100.jpg\n1 101 101.jpg\n",
        encoding="utf-8",
    )
    (meta_dir / "identity_CelebA.txt").write_text(
        "100.jpg 1234\n101.jpg 5678\n",
        encoding="utf-8",
    )

    out_manifest = tmp_path / "manifest.jsonl"
    items = build_baseline_manifest(
        ffhq_dir=ffhq_dir,
        celeba_dir=celeba_dir,
        metadata_dir=meta_dir,
        output_manifest=out_manifest,
    )

    assert len(items) == 4
    assert out_manifest.exists()
    assert all(item.label == 0 for item in items)
    assert all(item.data_source == "baseline" for item in items)
    assert items[0].source_real_id == "ffhq_00000"
    assert items[2].source_real_id == "celeba_id_1234"
