"""Isolated optional dependency profiles, preserving Kaggle's CUDA torch pair."""

import subprocess
import sys
from pathlib import Path
from typing import TextIO

from src.data.sd_demo_data import write_json

COMMON = [
    "diffusers==0.35.1",
    "transformers==4.56.2",
    "accelerate==1.10.1",
    "huggingface-hub==0.36.0",
    "safetensors==0.6.2",
    "peft==0.17.1",
    "sentencepiece>=0.2,<0.3",
    "protobuf>=3.20,<6",
    "einops",
    "scipy",
    "omegaconf",
    "ftfy",
    "regex",
]
FACE = ["insightface==0.7.3", "onnxruntime>=1.18,<2"]
PROFILES = {
    "controlnet": [],
    "ip_adapter": [],
    "faceid_sd15": FACE,
    "faceid_plusv2_xl": FACE,
    "instantid": FACE,
    "photomaker": FACE,
    "arc2face": FACE,
    "face_adapter": FACE
    + ["kornia", "matplotlib", "controlnet-aux", "spaces==0.19.4", "peft==0.9.0"],
    "reface": [
        "pytorch-lightning==1.9.5",
        "torchmetrics<1",
        "taming-transformers-rom1504",
        "kornia",
        "openai-clip",
        "albumentations<2",
        "dlib",
        "face-alignment==1.4.1",
        "invisible-watermark",
        "matplotlib",
        "natsort",
        "pudb",
        "imageio",
        "wandb",
    ],
    "pulid_v1": FACE + ["basicsr==1.4.2", "facexlib", "timm", "yacs", "k-diffusion"],
    "pulid_v11": FACE + ["basicsr==1.4.2", "facexlib", "timm", "yacs", "k-diffusion"],
    "fastcomposer": ["datasets<3", "evaluate", "openai-clip", "numpy<2"],
    "flashface": ["gradio>=3.41,<4", "easydict", "open-clip-torch", "timm"],
    "consistentid": FACE + ["einops", "sentencepiece"],
}


def prepare_environment(
    backend: str,
    env_root: Path,
    torch_versions: dict[str, str],
    *,
    log: TextIO | None = None,
    timeout: int = 3600,
) -> Path:
    """Create a venv per backend, constrained to the installed CUDA torch pair.

    Installation conflicts fail visibly; the user's global runtime is untouched.
    Downloaded source/checkpoint availability is tested by the worker, not here.
    """
    if backend == "diffusers":
        return Path(sys.executable)
    if backend not in PROFILES:
        raise ValueError(f"No environment profile for {backend}")
    root = env_root / backend
    python = root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    marker = root / "installed.json"
    requirements = COMMON + PROFILES[backend]
    # Older upstream APIs need their own Diffusers/Hub versions. CUDA torch stays
    # inherited from Kaggle; an incompatible dependency fails in this venv only.
    if backend in {"face_adapter", "fastcomposer"}:
        requirements = [
            r
            for r in COMMON
            if r.split("==")[0]
            not in {
                "diffusers",
                "transformers",
                "accelerate",
                "huggingface-hub",
                "peft",
            }
        ]
        requirements += (
            [
                "diffusers==0.26.0",
                "transformers==4.37.1",
                "accelerate==0.27.2",
                "huggingface-hub==0.25.2",
            ]
            if backend == "face_adapter"
            else [
                "diffusers==0.16.1",
                "transformers==4.25.1",
                "accelerate==0.19.0",
                "huggingface-hub==0.25.2",
            ]
        )
        requirements += PROFILES[backend]
    signature = {"requirements": requirements, "torch": torch_versions}
    if marker.is_file():
        import json

        if json.loads(marker.read_text()) == signature:
            return python
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(root)],
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    constraints = root / "cuda_constraints.txt"
    constraints.write_text(
        "".join(f"{name}=={version}\n" for name, version in torch_versions.items()),
        encoding="utf-8",
    )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-c", str(constraints), *requirements],
        check=True,
        stdout=log,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if backend == "fastcomposer":
        # The upstream package declares a legacy torch cap. Do not replace
        # Kaggle's CUDA pair to satisfy it; actual API compatibility is checked
        # by the worker and remains an explicit experimental profile.
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "facenet-pytorch==2.6.0",
            ],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    write_json(marker, signature)
    return python
