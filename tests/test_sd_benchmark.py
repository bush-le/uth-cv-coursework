"""Queue integrity, interruption and portable-result regression checks."""

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.data.sd_demo_data import file_sha256, write_json
from src.pipelines.sd_all_benchmark import (
    _error_kind,
    load_suite,
    queue_lock,
    run_suite,
    run_worker,
    valid_completed_artifacts,
)
from src.pipelines.sd_benchmark_archive import export_results, restore_results
from src.pipelines.sd_benchmark_env import COMMON, prepare_environment


def test_native_installs_are_scoped_to_venv_and_keep_kaggle_protobuf(
    tmp_path, monkeypatch
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "venv" in command:
            Path(command[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    python = prepare_environment(
        "diffusers", tmp_path, {"torch": "2.8.0", "torchvision": "0.23.0"}
    )
    assert python != Path(sys.executable)
    assert "--system-site-packages" in calls[0]
    assert all(command[0] == str(python) for command in calls if "pip" in command)
    assert "protobuf>=3.20" in COMMON
    assert not any("protobuf" in item and "<" in item for item in COMMON)
    prepare_environment(
        "diffusers", tmp_path, {"torch": "2.8.0", "torchvision": "0.23.0"}
    )
    assert len(calls) == 2  # Matching marker reuses the environment.


def test_kaggle_notebooks_never_install_inference_stack_into_kernel():
    for name in ("SD_IMAGE_DEMO_KAGGLE.ipynb", "SD_ALL_BENCHMARK_KAGGLE.ipynb"):
        notebook = json.loads((Path("notebooks") / name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "%pip install -q -r requirements/notebook.txt" in source
        assert "%pip install -q -r requirements/generation.txt" not in source


def output_fixture(root: Path) -> Path:
    directory = root / "cases" / "example"
    directory.mkdir(parents=True)
    image = directory / "result.png"
    Image.new("RGB", (16, 16), "red").save(image)
    record = {
        "image_id": "example",
        "file_path": image.as_posix(),
        "is_mock": False,
        "output_sha256": file_sha256(image),
    }
    (directory / "generated_manifest.jsonl").write_text(json.dumps(record) + "\n")
    write_json(directory / "summary.json", {"images": 1, "mock_outputs": 0})
    write_json(
        root / "queue.json",
        {"jobs": [{"directory": str(directory), "status": "completed"}]},
    )
    return directory


@pytest.mark.smoke
def test_catalog_is_covered_without_counting_frameworks_as_generators():
    suite = load_suite(Path("configs/generation/sd_all_benchmark.yaml"))
    assert len(suite["cases"]) == 32
    assert len(suite["frameworks"]) == 3
    assert sum(len(c["modes"]) for c in suite["cases"]) == 44
    assert {c["group"] for c in suite["cases"]} == {"checkpoints", "methods"}


def test_resume_rejects_missing_partial_mock_and_tampered_results(tmp_path):
    directory = output_fixture(tmp_path)
    assert valid_completed_artifacts(directory, 1)
    assert not valid_completed_artifacts(directory, 2)
    write_json(directory / "summary.json", {"images": 1, "mock_outputs": 1})
    assert not valid_completed_artifacts(directory)
    write_json(directory / "summary.json", {"images": 1, "mock_outputs": 0})
    Image.new("RGB", (16, 16), "blue").save(directory / "result.png")
    assert not valid_completed_artifacts(directory)
    (directory / "result.png").unlink()
    assert not valid_completed_artifacts(directory)


def test_restore_rebases_results_without_weights_or_overwrite(tmp_path):
    original = tmp_path / "original"
    output_fixture(original)
    for name in (
        "environments",
        "vendor",
        "cases/example/_model_cache",
        "cases/example/_runtime",
    ):
        folder = original / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "large_weight.bin").write_bytes(b"weights")
    archive = export_results(original, tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as stream:
        assert not any("large_weight" in name for name in stream.namelist())
    restored = restore_results(archive, tmp_path / "different_workspace")
    assert valid_completed_artifacts(restored / "cases/example", 1)
    queue = json.loads((restored / "queue.json").read_text())
    assert Path(queue["jobs"][0]["directory"]) == restored / "cases/example"
    with pytest.raises(FileExistsError):
        restore_results(archive, restored)


def test_restore_rejects_zip_traversal_before_extracting(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            "benchmark_archive.json",
            json.dumps({"schema_version": 1, "original_root": "/old"}),
        )
        stream.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        restore_results(archive, tmp_path / "new")
    assert not (tmp_path / "escaped.txt").exists()


def test_queue_lock_is_released_after_exception(tmp_path):
    with pytest.raises(ValueError):
        with queue_lock(tmp_path):
            with pytest.raises(RuntimeError, match="Another runner"):
                with queue_lock(tmp_path):
                    pytest.fail("Concurrent writer acquired lock")
            raise ValueError("Interrupted notebook")
    with queue_lock(tmp_path):
        assert True


def test_plan_does_not_start_gpu_and_preserves_interrupted_state(tmp_path, monkeypatch):
    suite = yaml.safe_load(Path("configs/generation/sd_all_benchmark.yaml").read_text())
    suite["run_dir"] = str(tmp_path / "run")
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite))
    state = run_suite(path, plan_only=True)
    state["jobs"][0]["status"] = "running"
    write_json(tmp_path / "run/queue.json", state)
    resumed = run_suite(path, plan_only=True)
    assert resumed["jobs"][0]["status"] == "interrupted"
    assert all(job["status"] != "completed" for job in resumed["jobs"])
    with pytest.raises(ValueError, match="Unknown"):
        run_suite(path, plan_only=True, case_ids=["made_up"])


def test_worker_timeout_and_nonzero_exit_are_not_success(tmp_path):
    with (tmp_path / "worker.log").open("w") as log:
        assert (
            run_worker(
                [sys.executable, "-c", "raise SystemExit(7)"], log, os.environ.copy(), 5
            )
            == 7
        )
        with pytest.raises(subprocess.TimeoutExpired):
            run_worker(
                [sys.executable, "-c", "import time; time.sleep(20)"],
                log,
                os.environ.copy(),
                0.1,
            )
    assert _error_kind("CUDA out of memory", 1) == "resource_failure"
    assert _error_kind("ImportError: x", 1) == "dependency_failure"
    assert _error_kind("401 Client Error", 1) == "access_required"


def test_queue_batches_skip_verified_outputs_and_retry_only_when_requested(
    tmp_path, monkeypatch
):
    import torch

    import src.pipelines.sd_all_benchmark as queue

    source = tmp_path / "real.png"
    Image.new("RGB", (32, 32), "green").save(source)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps({"image_id": "source", "file_path": str(source)}) + "\n"
    )
    pilot = yaml.safe_load(
        Path("configs/generation/kaggle_image_demo.yaml").read_text()
    )
    pilot["data"]["input_manifest"] = str(manifest)
    pilot_path = tmp_path / "pilot.yaml"
    pilot_path.write_text(yaml.safe_dump(pilot))
    suite = yaml.safe_load(Path("configs/generation/sd_all_benchmark.yaml").read_text())
    suite.update(run_dir=str(tmp_path / "run"), pilot_config=str(pilot_path))
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump(suite))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        queue, "prepare_environment", lambda *args, **kwargs: Path(sys.executable)
    )
    attempts = []
    failure = {"enabled": True}

    def worker(command, log, environment, timeout):
        payload = json.loads(Path(command[-1]).read_text())
        attempts.append(payload["case"]["id"])
        if failure["enabled"]:
            log.write("ImportError: deliberate fixture failure\n")
            return 1
        directory = Path(payload["run_dir"])
        records = []
        for index, mode in enumerate(payload["case"]["modes"]):
            path = directory / f"test_{index}.png"
            Image.new("RGB", (16, 16), "red").save(path)
            records.append(
                {
                    "image_id": mode,
                    "file_path": str(path),
                    "is_mock": False,
                    "output_sha256": file_sha256(path),
                }
            )
        (directory / "generated_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
        write_json(
            directory / "summary.json", {"images": len(records), "mock_outputs": 0}
        )
        return 0

    monkeypatch.setattr(queue, "run_worker", worker)
    first = run_suite(suite_path, case_ids=["sd15"], max_jobs=1)
    assert (
        next(j for j in first["jobs"] if j["case_id"] == "sd15")["failure_kind"]
        == "dependency_failure"
    )
    failure["enabled"] = False
    run_suite(suite_path, case_ids=["sd15"], max_jobs=1)
    assert len(attempts) == 1  # Failed job is retained until explicit retry.
    second = run_suite(suite_path, case_ids=["sd15"], max_jobs=1, retry_failed=True)
    assert (
        next(j for j in second["jobs"] if j["case_id"] == "sd15")["status"]
        == "completed"
    )
    run_suite(suite_path, case_ids=["sd15"], max_jobs=1)
    assert len(attempts) == 2
    manifest_record = next((tmp_path / "run/cases").rglob("generated_manifest.jsonl"))
    damaged_image = Path(
        json.loads(manifest_record.read_text().splitlines()[0])["file_path"]
    )
    damaged_image.write_bytes(b"corrupt")
    run_suite(suite_path, case_ids=["sd15"], max_jobs=1)
    assert len(attempts) == 3  # Corrupted completed output is regenerated.
