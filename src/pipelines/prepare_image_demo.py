"""Prepare a small real-image pilot from the user's public Drive folder."""

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from src.common.logger import get_logger
from src.data.sd_demo_data import file_sha256, read_manifest, write_json

logger = get_logger(__name__)


def list_public_drive(folder_id: str) -> dict[str, str]:
    """Return visible name->ID entries from a public folder listing.

    This intentionally inspects one public listing, not all paginated files.
    Missing selected files raise an error; no credentials or hidden APIs used.
    """
    from bs4 import BeautifulSoup

    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder_id):
        raise ValueError("Invalid Drive folder ID.")
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    with urllib.request.urlopen(url, timeout=30) as response:
        soup = BeautifulSoup(response.read(), "html.parser")
    entries = {}
    for node in soup.select("[data-id][data-tooltip]"):
        name = node.get_text(" ", strip=True)
        if name and "/" not in name and "\\" not in name:
            entries[name] = node["data-id"]
    if not entries:
        raise RuntimeError(
            "Drive listing empty/private or layout changed. Use local Kaggle input files instead."
        )
    return entries


def download_drive_file(file_id: str, destination: Path) -> None:
    """Download a public file atomically; gdown handles Drive confirmation."""
    import gdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    result = gdown.download(id=file_id, output=str(temporary), quiet=False)
    if not result or not temporary.is_file():
        raise RuntimeError(f"Drive download failed: {file_id}")
    temporary.replace(destination)


def prepare_drive_demo(
    folder_id: str,
    output_manifest: Path,
    download_root: Path,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Download only a few adult FFHQ face crops plus the small source manifest.

    The public manifest's age metadata is used for pilot selection only, not
    as verified ground truth. The result is demo_only, never a train/test split.
    """
    if not 1 <= limit <= 10:
        raise ValueError("Pilot limit must be 1..10; bulk data ingestion is separate.")
    listing = list_public_drive(folder_id)
    required = {"faces", "manifest.jsonl"}
    if not required.issubset(listing):
        raise ValueError("Expected faces/ and manifest.jsonl in Drive root.")
    full_manifest = download_root / "drive_manifest.jsonl"
    if not full_manifest.is_file():
        download_drive_file(listing["manifest.jsonl"], full_manifest)
    records = read_manifest(full_manifest)
    face_folders = list_public_drive(listing["faces"])
    ffhq = list_public_drive(face_folders["ffhq"])
    adult_groups = {
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80-89",
        "90-99",
    }
    selected = [
        r
        for r in records
        if int(r.get("label", -1)) == 0
        and r.get("demographics", {}).get("dataset") == "ffhq"
        and r.get("demographics", {}).get("age_group") in adult_groups
        and Path(r["file_path"]).name in ffhq
    ][:limit]
    if len(selected) != limit:
        raise RuntimeError(
            "Public listing has too few matching adult crops. Attach local data and prepare its manifest."
        )
    output = []
    for record in selected:
        name = Path(record["file_path"]).name
        destination = download_root / "faces" / "ffhq" / name
        if not destination.is_file():
            download_drive_file(ffhq[name], destination)
        with Image.open(destination) as image:
            image.verify()
        item = record | {
            "file_path": destination.as_posix(),
            "split": "demo_only",
            "drive_file_id": ffhq[name],
            "source_sha256": file_sha256(destination),
        }
        output.append(item)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output),
        encoding="utf-8",
    )
    write_json(
        output_manifest.with_suffix(".inventory.json"),
        {
            "drive_folder_id": folder_id,
            "source_manifest_sha256": file_sha256(full_manifest),
            "source_records": len(records),
            "selected": len(output),
            "checkpoint_found_in_root_listing": any(
                n.lower().endswith((".pt", ".pth", ".ckpt", ".safetensors"))
                for n in listing
            ),
            "selection": "adult FFHQ metadata, public listing subset, not representative benchmark",
        },
    )
    logger.info("Prepared %d real input images in %s", len(output), output_manifest)
    return output


def main() -> None:
    """Prepare sources without loading any Stable Diffusion weights."""
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generation/kaggle_image_demo.yaml")
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    prepare_drive_demo(
        data["drive_folder_id"],
        Path(data["input_manifest"]),
        Path(data["download_root"]),
        data["limit"],
    )


if __name__ == "__main__":
    main()
