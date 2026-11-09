"""Run the single-model pilot outside the notebook's shared Python kernel."""

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from src.pipelines.sd_benchmark_env import prepare_environment


def run_isolated_demo(config_path: Path) -> dict[str, Any]:
    """Launch real image inference in a venv; return the saved summary only.

    All model/tensor operations stay in the existing demo worker. The controller
    never imports Diffusers/Transformers or changes the kernel's dependencies.
    """
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["run_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    versions = {
        name: importlib.metadata.version(name) for name in ("torch", "torchvision")
    }
    environment = os.environ.copy()
    environment.update(USE_TF="0", USE_FLAX="0")
    environment["PYTHONPATH"] = (
        str(Path.cwd().resolve()) + os.pathsep + environment.get("PYTHONPATH", "")
    )
    with (output / "worker.log").open("w", encoding="utf-8") as log:
        python = prepare_environment(
            "diffusers", output / "environments", versions, log=log
        )
        subprocess.run(
            [
                str(python),
                "-m",
                "src.pipelines.sd_image_demo",
                "--config",
                str(config_path),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads((output / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    """CLI using the same isolated worker as the pilot notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/generation/kaggle_image_demo.yaml")
    )
    run_isolated_demo(parser.parse_args().config)


if __name__ == "__main__":
    main()
