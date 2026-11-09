"""Portable queue/results ZIPs, excluding environments and downloaded weights."""

import json
import zipfile
from pathlib import Path
from typing import Any

import yaml

from src.data.sd_demo_data import write_json

EXCLUDED = {"environments", "vendor", "_model_cache", "_runtime", "__pycache__"}
MARKER = "benchmark_archive.json"


def export_results(root: Path, destination: Path) -> Path:
    """Export queue and image artifacts; source data must be prepared separately."""
    root, destination = root.resolve(), destination.resolve()
    if not (root / "queue.json").is_file():
        raise FileNotFoundError("No queue.json to export.")
    if destination.is_relative_to(root):
        raise ValueError("Write the ZIP outside the run directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            MARKER, json.dumps({"schema_version": 1, "original_root": str(root)})
        )
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (
                path.name != "queue.lock"
                and path.is_file()
                and not path.is_symlink()
                and not EXCLUDED.intersection(relative.parts)
            ):
                archive.write(path, relative.as_posix())
    return destination


def _rebase(value: Any, old: str, new: str) -> Any:
    """Rebase only absolute metadata paths within the exported run directory."""
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        original = old.replace("\\", "/").rstrip("/")
        if normalized == original or normalized.startswith(original + "/"):
            return new.rstrip("/") + normalized[len(original) :]
        return value
    if isinstance(value, list):
        return [_rebase(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _rebase(item, old, new) for key, item in value.items()}
    return value


def restore_results(archive_path: Path, root: Path) -> Path:
    """Restore into an empty directory, rejecting traversal and symlink entries.

    Rebase result paths across Kaggle workspaces. Model weights/environments are
    installed again. Reuse the same source bundle and original input images;
    otherwise the queue fingerprint deliberately schedules new jobs.
    """
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Restore destination must be empty; existing results are never overwritten."
        )
    with zipfile.ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read(MARKER))
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported benchmark archive.")
        for info in archive.infolist():
            target = (root / info.filename).resolve()
            if (
                not target.is_relative_to(root)
                or "\\" in info.filename
                or (info.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"Unsafe archive entry: {info.filename}")
        root.mkdir(parents=True, exist_ok=True)
        archive.extractall(root)
    old = metadata["original_root"]
    for path in root.rglob("*"):
        if path.suffix == ".json":
            write_json(
                path,
                _rebase(
                    json.loads(path.read_text(encoding="utf-8")), old, root.as_posix()
                ),
            )
        elif path.suffix == ".jsonl":
            records = [
                _rebase(json.loads(line), old, root.as_posix())
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
        elif path.suffix in {".yaml", ".yml"}:
            path.write_text(
                yaml.safe_dump(
                    _rebase(
                        yaml.safe_load(path.read_text(encoding="utf-8")),
                        old,
                        root.as_posix(),
                    )
                ),
                encoding="utf-8",
            )
    return root
