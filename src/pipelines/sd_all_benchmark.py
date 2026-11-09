"""Sequential Kaggle queue covering the entire SD reference catalog."""

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from src.common.logger import get_logger
from src.data.sd_demo_data import (
    file_sha256,
    read_manifest,
    resolve_image_path,
    write_json,
)
from src.pipelines.sd_benchmark_env import prepare_environment

logger = get_logger(__name__)


@contextmanager
def queue_lock(root: Path) -> Iterator[None]:
    """OS-released exclusive lock: an interrupted notebook cannot leave it held."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another runner is using this queue.") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another runner is using this queue.") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def load_suite(path: Path) -> dict[str, Any]:
    """Validate complete catalog coverage and unique case IDs before downloads."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    catalog = yaml.safe_load(Path(config["catalog"]).read_text(encoding="utf-8"))
    ids = [case["id"] for case in config["cases"]]
    if len(ids) != len(set(ids)) or any(not x.replace("_", "").isalnum() for x in ids):
        raise ValueError("Case IDs must be unique filesystem-safe identifiers.")
    covered = {case["catalog_id"] for case in config["cases"]} | {
        f["catalog_id"] for f in config["frameworks"]
    }
    expected = {entry["id"] for entry in catalog["entries"]}
    if covered != expected:
        raise ValueError(
            f"Catalog coverage mismatch: missing={expected - covered}; extra={covered - expected}"
        )
    for case in config["cases"]:
        if case["group"] not in {"checkpoints", "methods"}:
            raise ValueError("Generation jobs must be checkpoints or methods.")
        if not case["modes"] or case["steps"] < 1 or case["size"] % 8:
            raise ValueError(f"Invalid generation contract: {case['id']}")
        if case["backend"] != "diffusers" and case["modes"] != ["reference"]:
            raise ValueError("Reference backends must use their actual reference task.")
    return config


def valid_completed_artifacts(
    directory: Path, expected_count: int | None = None
) -> bool:
    """Resume only outputs with matching hashes, successful decode and no mocks."""
    from PIL import Image

    try:
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        records = read_manifest(directory / "generated_manifest.jsonl")
        if (
            not records
            or summary.get("mock_outputs") != 0
            or summary.get("images") != len(records)
        ):
            return False
        if expected_count is not None and len(records) != expected_count:
            return False
        for record in records:
            if record.get("is_mock") is not False:
                return False
            path = Path(record["file_path"])
            expected = record.get("output_sha256")
            if expected is None:
                expected = json.loads(path.with_suffix(".json").read_text())[
                    "output_sha256"
                ]
            if file_sha256(path) != expected:
                return False
            with Image.open(path) as image:
                image.verify()
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_worker(
    command: list[str], log: Any, environment: dict[str, str], timeout: int
) -> int:
    """Terminate the entire Linux worker group, including upstream CLI children."""
    with subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=environment,
        start_new_session=os.name != "nt",
    ) as process:
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Worker exited during the interruption race.
            else:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            process.wait()
            raise


def _error_kind(log: str, exit_code: int | None) -> str:
    """Classify failures without interpreting them as model quality scores."""
    lowered = log.lower()
    if "out of memory" in lowered:
        return "resource_failure"
    if "gatedrepo" in lowered or "401 client" in lowered or "403 client" in lowered:
        return "access_required"
    if (
        "modulenotfound" in lowered
        or "importerror" in lowered
        or "resolutionimpossible" in lowered
    ):
        return "dependency_failure"
    if exit_code is None:
        return "timeout"
    return "execution_failure"


def write_dashboard(state: dict[str, Any], output: Path) -> None:
    """Export a readable queue summary; never invent quality/ranking values."""
    rows = "".join(
        f"<tr><td>{html.escape(j['case_id'])}</td><td>{html.escape(j['group'])}</td><td>{html.escape(j['status'])}</td><td>{html.escape(j.get('failure_kind', ''))}</td><td>{j.get('wall_seconds', '')}</td></tr>"
        for j in state["jobs"]
    )
    tools = ", ".join(f["name"] for f in state["frameworks"])
    output.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SD benchmark queue</title><style>body{font:15px/1.6 Segoe UI,Arial;margin:30px;background:#f6f6f1;color:#17242e}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:12px;text-align:left}th{background:#edf2fc}</style><h1>SD benchmark queue</h1><p>Completed means real outputs validated, not that identity or quality passed. Failed jobs are not assigned quality scores.</p><p>Framework inventory (not independent image methods): '
        + html.escape(tools)
        + "</p><table><tr><th>Case</th><th>Group</th><th>Status</th><th>Failure kind</th><th>Wall seconds</th></tr>"
        + rows
        + "</table></html>",
        encoding="utf-8",
    )


def run_suite(config_path: Path, **options: Any) -> dict[str, Any]:
    """Lock the suite before inspecting or changing its persistent queue."""
    config = load_suite(config_path)
    with queue_lock(Path(config["run_dir"]).resolve()):
        return _run_suite(config_path, **options)


def _run_suite(
    config_path: Path,
    *,
    group: str = "all",
    plan_only: bool = False,
    max_jobs: int | None = None,
    retry_failed: bool = False,
    keep_model_cache: bool = False,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run one GPU job at a time with process isolation and persistent state.

    Checkpoints use native mode-specific pipelines. Reference methods use
    actual public upstream implementations. Failed/gated/OOM cases remain in
    the report. Framework names are retained as tools, never fake generators.
    """
    config = load_suite(config_path)
    if case_ids and set(case_ids) - {case["id"] for case in config["cases"]}:
        raise ValueError("Unknown case ID requested.")
    if group not in {"all", "checkpoints", "methods"}:
        raise ValueError("group must be all, checkpoints or methods")
    if max_jobs is not None and max_jobs < 1:
        raise ValueError("max_jobs must be positive.")
    project_root = Path.cwd().resolve()
    root = Path(config["run_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "queue.json"
    previous = (
        json.loads(state_path.read_text()) if state_path.is_file() else {"jobs": []}
    )
    old_jobs = {job["case_id"]: job for job in previous["jobs"]}
    pilot = yaml.safe_load(Path(config["pilot_config"]).read_text(encoding="utf-8"))
    input_manifest = Path(pilot["data"]["input_manifest"]).resolve()
    inputs = read_manifest(input_manifest) if input_manifest.is_file() else []
    if not inputs and not plan_only:
        raise FileNotFoundError(
            "Prepare inputs first with src.pipelines.prepare_image_demo."
        )
    input_hashes = [
        {
            "id": r["image_id"],
            "sha256": file_sha256(resolve_image_path(r["file_path"], input_manifest)),
        }
        for r in inputs
    ]
    code_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("src").rglob("*.py")))
    ).hexdigest()
    jobs = []
    if config["source_limit"] < 1 or config["timeout_seconds"] < 1:
        raise ValueError("source_limit and timeout_seconds must be positive.")
    for case in config["cases"]:
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "case": case,
                    "inputs": input_hashes,
                    "pilot": pilot,
                    "source_limit": config["source_limit"],
                    "code_hash": code_hash,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        old = old_jobs.get(case["id"], {})
        matching = old.get("fingerprint") == fingerprint
        status = old.get("status", "pending") if matching else "pending"
        if status == "running":
            status = "interrupted"
        job = {
            "case_id": case["id"],
            "catalog_id": case["catalog_id"],
            "group": case["group"],
            "fingerprint": fingerprint,
            "status": status,
            "directory": str(root / "cases" / f"{case['id']}_{fingerprint[:10]}"),
        }
        if matching:
            job.update({k: v for k, v in old.items() if k not in job})
        jobs.append(job)
    state = {
        "schema_version": 1,
        "jobs": jobs,
        "frameworks": config["frameworks"],
        "catalog_entries": len(
            yaml.safe_load(Path(config["catalog"]).read_text(encoding="utf-8"))[
                "entries"
            ]
        ),
        "variant_cases": len(jobs),
        "full_quality_evaluation": False,
    }

    def save() -> None:
        write_json(state_path, state)
        write_dashboard(state, root / "index.html")

    save()
    if plan_only:
        logger.info("Prepared %d generation cases without model downloads.", len(jobs))
        return state
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle GPU. Plan is saved; no mock tasks were run.")
    attempts = 0
    for case, job in zip(config["cases"], jobs, strict=True):
        if case_ids and case["id"] not in case_ids:
            continue
        if group != "all" and case["group"] != group:
            continue
        directory = Path(job["directory"])
        expected_count = min(len(inputs), config["source_limit"]) * (
            len(case["modes"]) * pilot["generation"]["variants_per_mode"]
            if case["backend"] == "diffusers"
            else 1
        )
        if job["status"] == "completed" and valid_completed_artifacts(
            directory, expected_count
        ):
            continue
        if job["status"] == "failed" and not retry_failed:
            continue
        if max_jobs is not None and attempts >= max_jobs:
            break
        attempts += 1
        directory.mkdir(parents=True, exist_ok=True)
        job.update(status="running", failure_kind="")
        save()
        start = time.perf_counter()
        log_path = directory / "worker.log"
        try:
            selected = inputs[: config["source_limit"]]
            selected_manifest = directory / "input_manifest.jsonl"
            selected_manifest.write_text(
                "".join(
                    json.dumps(
                        r
                        | {
                            "file_path": resolve_image_path(
                                r["file_path"], input_manifest
                            ).as_posix()
                        }
                    )
                    + "\n"
                    for r in selected
                ),
                encoding="utf-8",
            )
            worker_config = json.loads(json.dumps(pilot))
            worker_config.update(
                model_name=case["model_name"],
                run_dir=str(directory),
                device="cuda",
                require_cuda=True,
            )
            worker_config["data"]["input_manifest"] = str(selected_manifest)
            worker_config["generation"].update(
                height=case["size"], width=case["size"], modes=case["modes"]
            )
            worker_config["engine"].update(
                case["engine"],
                num_inference_steps=case["steps"],
                guidance_scale=case["guidance"],
            )
            worker_config["engine"]["inpaint_model_name"] = case["model_name"]
            (directory / "config.yaml").write_text(
                yaml.safe_dump(worker_config), encoding="utf-8"
            )
            payload = {
                "case": case,
                "run_dir": str(directory),
                "input_manifest": str(input_manifest),
                "vendor_root": str(root / "vendor"),
                "seed": pilot["generation"]["seed"],
                "prompt": pilot["generation"]["prompts"]["text2img"],
                "source_limit": config["source_limit"],
            }
            write_json(directory / "payload.json", payload)
            versions = {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision")
            }
            with log_path.open("w", encoding="utf-8") as log:
                python = prepare_environment(
                    case["backend"],
                    root / "environments",
                    versions,
                    log=log,
                    timeout=config["timeout_seconds"],
                )
                environment = os.environ.copy()
                environment["PYTHONPATH"] = (
                    str(project_root) + os.pathsep + environment.get("PYTHONPATH", "")
                )
                cache = directory / "_model_cache"
                environment["HF_HOME"] = str(cache)
                environment["HF_HUB_CACHE"] = str(cache / "hub")
                environment["HF_ASSETS_CACHE"] = str(cache / "assets")
                environment["TORCH_HOME"] = str(cache / "torch")
                environment["XDG_CACHE_HOME"] = str(cache / "xdg")
                environment.pop("TRANSFORMERS_CACHE", None)
                environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
                environment["USE_TF"] = "0"
                environment["USE_FLAX"] = "0"
                exit_code = run_worker(
                    [
                        str(python),
                        "-m",
                        "src.pipelines.sd_benchmark_worker",
                        "--payload",
                        str(directory / "payload.json"),
                    ],
                    log,
                    environment,
                    config["timeout_seconds"],
                )
            if exit_code or not valid_completed_artifacts(directory, expected_count):
                text = log_path.read_text(encoding="utf-8", errors="replace")
                job.update(
                    status="failed",
                    failure_kind=_error_kind(text, exit_code),
                    exit_code=exit_code,
                )
            else:
                job.update(status="completed", exit_code=0)
        except subprocess.TimeoutExpired:
            job.update(status="failed", failure_kind="timeout")
        except Exception as exc:
            log_text = (
                log_path.read_text(encoding="utf-8", errors="replace")
                if log_path.exists()
                else ""
            )
            category = _error_kind(log_text, 1)
            job.update(
                status="failed",
                failure_kind=category
                if category != "execution_failure"
                else "setup_failure",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            job["wall_seconds"] = round(time.perf_counter() - start, 3)
            job["log"] = str(log_path)
            save()
            if not keep_model_cache:
                # Only delete the suite's own per-case cache, never the user's HF cache.
                for cache_name in ("_model_cache", "_runtime"):
                    cache = (directory / cache_name).resolve()
                    if (
                        cache.is_dir()
                        and cache.is_relative_to(root / "cases")
                        and cache.name == cache_name
                    ):
                        shutil.rmtree(cache)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generation/sd_all_benchmark.yaml")
    )
    parser.add_argument(
        "--group", choices=["all", "checkpoints", "methods"], default="all"
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-model-cache", action="store_true")
    parser.add_argument(
        "--cases", nargs="+", help="Optional case IDs; other jobs remain queued."
    )
    args = parser.parse_args()
    run_suite(
        args.config,
        group=args.group,
        plan_only=args.plan_only,
        max_jobs=args.max_jobs,
        retry_failed=args.retry_failed,
        keep_model_cache=args.keep_model_cache,
        case_ids=args.cases,
    )


if __name__ == "__main__":
    main()
