"""Reference-conditioned SD backends using public upstream implementations.

Each worker has an isolated process/environment. Optional dependencies are
imported only when selected. Inputs are PIL RGB; NumPy bridges are RGB uint8
(H,W,3), with explicit BGR conversion for face analysis. No mock fallbacks.
"""

import ast
import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.sd_demo_data import write_json

REPOSITORIES = {
    "faceid_sd15": "https://github.com/tencent-ailab/IP-Adapter.git",
    "faceid_plusv2_xl": "https://github.com/tencent-ailab/IP-Adapter.git",
    "instantid": "https://github.com/instantX-research/InstantID.git",
    "photomaker": "https://github.com/TencentARC/PhotoMaker.git",
    "face_adapter": "https://github.com/FaceAdapter/Face-Adapter.git",
    "reface": "https://github.com/Sanoojan/REFace.git",
    "arc2face": "https://github.com/foivospar/Arc2Face.git",
    "pulid_v1": "https://github.com/ToTheBeginning/PuLID.git",
    "pulid_v11": "https://github.com/ToTheBeginning/PuLID.git",
    "fastcomposer": "https://github.com/mit-han-lab/fastcomposer.git",
    "flashface": "https://github.com/ali-vilab/FlashFace.git",
    "consistentid": "https://github.com/JackAILab/ConsistentID.git",
}


def checkout_backend(
    name: str, vendor_root: Path, revision: str | None = None
) -> Path | None:
    """Clone once and freeze the resulting commit; never pull on resume."""
    url = REPOSITORIES.get(name)
    if url is None:
        return None
    path = vendor_root / url.rsplit("/", 1)[-1].removesuffix(".git")
    if not (path / ".git").is_dir():
        subprocess.run(["git", "clone", "--depth", "1", url, str(path)], check=True)
    if revision:
        current = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        if current != revision:
            subprocess.run(
                ["git", "-C", str(path), "fetch", "--depth", "1", "origin", revision],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(path), "checkout", "--detach", revision], check=True
            )
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    write_json(
        path.parent / f"{path.name}.revision.json", {"url": url, "revision": revision}
    )
    return path.resolve()


def _face_analyzer(recognition_override: str | None = None) -> Any:
    """Load public InsightFace antelopev2 for research; no zero-vector fallback."""
    from huggingface_hub import snapshot_download
    from insightface.app import FaceAnalysis

    snapshot_download("DIAMONIK7777/antelopev2", local_dir="models/antelopev2")
    if recognition_override:
        shutil.copy2(recognition_override, "models/antelopev2/glintr100.onnx")
    app = FaceAnalysis(name="antelopev2", root=".", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def _face(app: Any, source: Image.Image) -> Any:
    """Pick largest detected face from RGB uint8 (H,W,3), converting to BGR."""
    faces = app.get(cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR))
    if not faces:
        raise RuntimeError(
            "No reference face detected; identity generation cannot proceed."
        )
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _take_image(result: Any) -> Image.Image:
    """Normalize an upstream result without hiding invalid or rejected outputs."""
    flags = getattr(result, "nsfw_content_detected", None)
    if flags is not None and any(flags):
        raise RuntimeError("Upstream safety checker rejected the sample.")
    if hasattr(result, "images"):
        result = result.images
    if isinstance(result, (list, tuple)):
        result = result[0]
    if isinstance(result, np.ndarray):
        if result.dtype != np.uint8:
            raise TypeError("Expected upstream uint8 RGB image, not unscaled features.")
        result = Image.fromarray(result)
    if not isinstance(result, Image.Image):
        raise TypeError(f"Unsupported upstream result: {type(result)}")
    return result.convert("RGB")


def load_reference_backend(
    case: dict[str, Any],
    repo: Path | None,
    runtime: Path,
) -> Callable[[Image.Image, Image.Image, str, int], Image.Image]:
    """Load selected public backend. Callable takes source, target, prompt, seed.

    Source and target are distinct for face swapping; personalization methods
    use the source only. Upstream-specific output sizes are retained and logged.
    """
    name = case["backend"]
    runtime.mkdir(parents=True, exist_ok=True)
    os.chdir(runtime)
    if name in {"face_adapter", "reface", "fastcomposer"}:
        if repo is None:
            raise ValueError(f"Missing upstream repository for {name}")
        return _cli_backend(case, repo, runtime)
    from diffusers import (
        AutoPipelineForText2Image,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
    )
    from huggingface_hub import hf_hub_download, snapshot_download

    if repo is not None:
        sys.path.insert(0, str(repo))
        for subfolder in ("src", "src/taming-transformers"):
            if (repo / subfolder).is_dir():
                sys.path.append(str(repo / subfolder))
    device = "cuda"
    dtype = torch.float16
    size, steps, guidance = case["size"], case["steps"], case["guidance"]
    model = case["model_name"]
    negative = "blurry, deformed, distorted face, poor anatomy, text, watermark"

    def kwargs(seed: int) -> dict[str, Any]:
        return {
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "height": size,
            "width": size,
            "negative_prompt": negative,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
        }

    if name == "controlnet":
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

        control = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny", torch_dtype=dtype
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model, controlnet=control, torch_dtype=dtype
        ).to(device)

        def generate(
            source: Image.Image, target: Image.Image, prompt: str, seed: int
        ) -> Image.Image:
            gray = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            condition = Image.fromarray(np.repeat(edges[..., None], 3, axis=2))
            return _take_image(pipe(prompt=prompt, image=condition, **kwargs(seed)))

        return generate

    if name == "ip_adapter":
        pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype).to(
            device
        )
        pipe.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
        )
        pipe.set_ip_adapter_scale(0.6)
        return lambda source, target, prompt, seed: _take_image(
            pipe(prompt=prompt, ip_adapter_image=source, **kwargs(seed))
        )

    if name in {"faceid_sd15", "faceid_plusv2_xl"}:
        from ip_adapter.ip_adapter_faceid import IPAdapterFaceID, IPAdapterFaceIDPlusXL

        app = _face_analyzer()
        pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype).to(
            device
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        filename = (
            "ip-adapter-faceid_sd15.bin"
            if name == "faceid_sd15"
            else "ip-adapter-faceid-plusv2_sdxl.bin"
        )
        ckpt = hf_hub_download("h94/IP-Adapter-FaceID", filename)
        adapter = (
            IPAdapterFaceID(pipe, ckpt, device)
            if name == "faceid_sd15"
            else IPAdapterFaceIDPlusXL(
                pipe, "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", ckpt, device
            )
        )

        def generate(
            source: Image.Image, target: Image.Image, prompt: str, seed: int
        ) -> Image.Image:
            face = _face(app, source)
            args = {
                "faceid_embeds": torch.from_numpy(face.normed_embedding).unsqueeze(0),
                "prompt": prompt,
                "negative_prompt": negative,
                "num_samples": 1,
                "seed": seed,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "width": size,
                "height": size,
            }
            if name == "faceid_plusv2_xl":
                from insightface.utils import face_align

                crop = face_align.norm_crop(
                    cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR),
                    landmark=face.kps,
                    image_size=224,
                )
                args["face_image"] = Image.fromarray(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                )
            return _take_image(adapter.generate(**args))

        return generate

    if name == "instantid":
        from diffusers import ControlNetModel
        from pipeline_stable_diffusion_xl_instantid import (
            StableDiffusionXLInstantIDPipeline,
            draw_kps,
        )

        app = _face_analyzer()
        control = ControlNetModel.from_pretrained(
            "InstantX/InstantID", subfolder="ControlNetModel", torch_dtype=dtype
        )
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            model, controlnet=control, torch_dtype=dtype
        ).to(device)
        pipe.load_ip_adapter_instantid(
            hf_hub_download("InstantX/InstantID", "ip-adapter.bin")
        )

        def generate(
            source: Image.Image, target: Image.Image, prompt: str, seed: int
        ) -> Image.Image:
            face = _face(app, source)
            return _take_image(
                pipe(
                    prompt=prompt,
                    image_embeds=face.embedding,
                    image=draw_kps(source, face.kps),
                    controlnet_conditioning_scale=0.8,
                    ip_adapter_scale=0.8,
                    **kwargs(seed),
                )
            )

        return generate

    if name == "photomaker":
        from photomaker import PhotoMakerStableDiffusionXLPipeline

        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            model, torch_dtype=dtype
        ).to(device)
        checkpoint = Path(hf_hub_download("TencentARC/PhotoMaker", "photomaker-v1.bin"))
        pipe.load_photomaker_adapter(
            str(checkpoint.parent),
            weight_name=checkpoint.name,
            subfolder="",
            trigger_word="img",
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()
        return lambda source, target, prompt, seed: _take_image(
            pipe(
                prompt="a photo of an adult person img, studio lighting, natural skin",
                input_id_images=[source],
                start_merge_step=min(10, steps - 1),
                **kwargs(seed),
            )
        )

    if name == "arc2face":
        from arc2face import CLIPTextModelWrapper, project_face_embs
        from diffusers import UNet2DConditionModel

        root = snapshot_download(
            "FoivosPar/Arc2Face", allow_patterns=["arc2face/*", "encoder/*"]
        )
        app = _face_analyzer(hf_hub_download("FoivosPar/Arc2Face", "arcface.onnx"))
        encoder = CLIPTextModelWrapper.from_pretrained(
            root, subfolder="encoder", torch_dtype=dtype
        )
        unet = UNet2DConditionModel.from_pretrained(
            root, subfolder="arc2face", torch_dtype=dtype
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model, text_encoder=encoder, unet=unet, torch_dtype=dtype
        ).to(device)

        def generate(
            source: Image.Image, target: Image.Image, prompt: str, seed: int
        ) -> Image.Image:
            identity = (
                torch.from_numpy(_face(app, source).embedding)
                .unsqueeze(0)
                .to(device, dtype)
            )
            identity = identity / torch.linalg.vector_norm(
                identity, dim=1, keepdim=True
            )
            embedded = project_face_embs(pipe, identity)
            return _take_image(pipe(prompt_embeds=embedded, **kwargs(seed)))

        return generate

    if name in {"pulid_v1", "pulid_v11"}:
        # BasicSR still imports the former torchvision module name.
        if importlib.util.find_spec("torchvision.transforms.functional_tensor") is None:
            sys.modules["torchvision.transforms.functional_tensor"] = (
                importlib.import_module("torchvision.transforms._functional_tensor")
            )
        module = importlib.import_module(
            "pulid.pipeline" if name == "pulid_v1" else "pulid.pipeline_v1_1"
        )
        adapter = (
            module.PuLIDPipeline()
            if name == "pulid_v1"
            else module.PuLIDPipeline(sdxl_repo=model, sampler="dpmpp_2m")
        )

        def generate(
            source: Image.Image, target: Image.Image, prompt: str, seed: int
        ) -> Image.Image:
            identity = adapter.get_id_embedding([np.asarray(source)])
            if name == "pulid_v1":
                return _take_image(
                    adapter.inference(
                        prompt,
                        (1, size, size),
                        negative,
                        identity,
                        0.8,
                        guidance,
                        steps,
                    )
                )
            uncond, cond = identity
            return _take_image(
                adapter.inference(
                    prompt,
                    (1, size, size),
                    negative,
                    cond,
                    uncond,
                    0.8,
                    guidance,
                    steps,
                    seed,
                )
            )

        return generate

    if name == "consistentid":
        from pipline_StableDiffusion_ConsistentID import (
            ConsistentIDStableDiffusionPipeline,
        )

        pipe = ConsistentIDStableDiffusionPipeline.from_pretrained(
            model, torch_dtype=dtype
        ).to(device)
        checkpoint = Path(
            hf_hub_download("JackAILab/ConsistentID", "ConsistentID-v1.bin")
        )
        parsing = hf_hub_download("JackAILab/ConsistentID", "face_parsing.pth")
        pipe.load_ConsistentID_model(
            str(checkpoint.parent),
            image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            bise_net_cp=parsing,
            weight_name=checkpoint.name,
            subfolder="",
            trigger_word="img",
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        return lambda source, target, prompt, seed: _take_image(
            pipe(
                prompt=prompt,
                input_id_images=source,
                start_merge_step=min(30, steps - 1),
                **kwargs(seed),
            )
        )

    if name == "flashface":
        assert repo is not None
        snapshot_download("shilongz/FlashFace-SD1.5", local_dir="cache")
        sys.path.insert(0, str(repo / "flashface/all_finetune"))
        source_file = repo / "flashface/all_finetune/demo_gradio.py"
        tree = ast.parse(source_file.read_text())
        kept = []
        for statement in tree.body:
            if isinstance(statement, ast.With) and any(
                "gr.Blocks" in ast.unparse(item.context_expr)
                for item in statement.items
            ):
                break  # retain model and inference code, do not launch a web server
            kept.append(statement)
        namespace: dict[str, Any] = {
            "__file__": str(source_file),
            "__name__": "flashface_benchmark",
        }
        exec(
            compile(ast.Module(body=kept, type_ignores=[]), str(source_file), "exec"),
            namespace,
        )
        return lambda source, target, prompt, seed: _take_image(
            namespace["generate"](
                prompt,
                negative,
                steps=steps,
                seed=seed,
                num_sample=1,
                reference_face_1=np.asarray(source),
                face_bbox=[0.3, 0.2, 0.7, 0.7],
            )
        )

    if name in {"face_adapter", "reface", "fastcomposer"}:
        assert repo is not None
        return _cli_backend(case, repo, runtime)
    raise ValueError(f"Unregistered reference backend: {name}")


def _cli_backend(case: dict[str, Any], repo: Path, runtime: Path) -> Callable:
    """Use the author's actual CLI instead of substituting generic inpainting."""
    from huggingface_hub import hf_hub_download, snapshot_download

    name = case["backend"]
    if name == "face_adapter":
        checkpoint = snapshot_download("FaceAdapter/FaceAdapter")
    elif name == "reface":
        checkpoint = hf_hub_download("Sanoojan/REFace", "last.ckpt")
        snapshot_download(
            "Sanoojan/REFace",
            local_dir=str(repo),
            allow_patterns=[
                "Other_dependencies/DLIB_landmark_det/*.dat",
                "Other_dependencies/arcface/*.pth",
                "Other_dependencies/face_parsing/*.pth",
            ],
        )
    else:
        checkpoint = str(
            Path(
                hf_hub_download("mit-han-lab/fastcomposer", "pytorch_model.bin")
            ).parent
        )
    count = 0

    def generate(
        source: Image.Image, target: Image.Image, prompt: str, seed: int
    ) -> Image.Image:
        nonlocal count
        count += 1
        work = runtime / f"input_{count}"
        src, tgt, out = work / "source", work / "target", work / "output"
        src.mkdir(parents=True, exist_ok=True)
        tgt.mkdir(parents=True, exist_ok=True)
        source.save(src / "source.png")
        target.save(tgt / "target.png")
        if name == "face_adapter":
            command = [
                sys.executable,
                str(repo / "infer.py"),
                "--source",
                str(src),
                "--target",
                str(tgt),
                "--output",
                str(out),
                "--checkpoint",
                checkpoint,
                "--base_model",
                case["model_name"],
            ]
            pattern = "swap/*.png"
        elif name == "reface":
            config = repo / "configs/project_ffhq.yaml"
            if not config.is_file():
                candidates = list(repo.rglob("project_ffhq.yaml"))
                if not candidates:
                    raise FileNotFoundError(
                        "Upstream REFace project_ffhq.yaml unavailable; no config fabricated."
                    )
                config = candidates[0]
            command = [
                sys.executable,
                str(repo / "scripts/inference_swap_selected.py"),
                "--outdir",
                str(out),
                "--target_folder",
                str(tgt),
                "--src_folder",
                str(src),
                "--config",
                str(config),
                "--ckpt",
                checkpoint,
                "--Base_dir",
                str(work / "base"),
                "--n_samples",
                "1",
                "--scale",
                "3.5",
                "--ddim_steps",
                str(case["steps"]),
            ]
            pattern = "results/*/*.png"
        else:
            command = [
                sys.executable,
                str(repo / "fastcomposer/inference.py"),
                "--pretrained_model_name_or_path",
                case["model_name"],
                "--finetuned_model_path",
                checkpoint,
                "--test_reference_folder",
                str(src),
                "--test_caption",
                "an adult person <|image|> in studio lighting",
                "--output_dir",
                str(out),
                "--mixed_precision",
                "fp16",
                "--image_encoder_type",
                "clip",
                "--image_encoder_name_or_path",
                "openai/clip-vit-large-patch14",
                "--num_image_tokens",
                "1",
                "--max_num_objects",
                "1",
                "--object_resolution",
                "224",
                "--generate_height",
                str(case["size"]),
                "--generate_width",
                str(case["size"]),
                "--num_images_per_prompt",
                "1",
                "--num_rows",
                "1",
                "--seed",
                str(seed),
                "--guidance_scale",
                str(case["guidance"]),
                "--inference_steps",
                str(case["steps"]),
                "--start_merge_step",
                "10",
                "--no_object_augmentation",
            ]
            pattern = "output_0.png"
        environment = os.environ.copy()
        # REFace also has a src namespace. It must resolve its own modules in
        # this subprocess rather than importing the coursework's src package.
        environment["PYTHONPATH"] = str(repo)
        subprocess.run(command, cwd=repo, env=environment, check=True)
        images = sorted(out.glob(pattern))
        if len(images) != 1:
            raise RuntimeError(
                f"Upstream output contract ambiguous: {len(images)} candidates in {out}; inspect log, no image chosen silently."
            )
        with Image.open(images[0]) as image:
            return image.convert("RGB").copy()

    return generate
