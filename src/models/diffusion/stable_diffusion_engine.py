"""Native SD pipelines. Mock mode is explicit and is never a runtime fallback."""

import gc
import hashlib
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.common.logger import get_logger

logger = get_logger(__name__)
DEFAULT_SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def validate_rgb(image: torch.Tensor) -> None:
    """Require finite RGB floating (3,H,W) in [0,1], on any device."""
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("Expected RGB floating tensor (3,H,W).")
    if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("RGB values must be finite and in [0,1].")


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert RGB float (3,H,W) [0,1] on any device to PIL RGB (W,H)."""
    validate_rgb(image)
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8))


class StableDiffusionEngine(nn.Module):
    """One resident pipeline; outputs CPU float32 RGB (3,H,W) in [0,1]."""

    def __init__(
        self,
        model_name: str = DEFAULT_SD,
        device: str | torch.device | None = None,
        *,
        inpaint_model_name: str | None = None,
        revision: str | None = None,
        inpaint_revision: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        negative_prompt: str = "deformed, blurry face",
        cpu_offload: bool = False,
        local_files_only: bool = False,
        pipeline_family: str = "sd1",
        scheduler: str | None = None,
        timestep_spacing: str | None = None,
        unet_override: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.inpaint_model_name = inpaint_model_name or model_name
        self.target_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if num_inference_steps < 1 or guidance_scale < 0:
            raise ValueError("Invalid steps or guidance_scale.")
        if self.target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable. Enable a Kaggle GPU or choose cpu explicitly."
            )
        self.revision, self.inpaint_revision = revision, inpaint_revision
        self.num_inference_steps, self.guidance_scale = (
            num_inference_steps,
            guidance_scale,
        )
        self.negative_prompt = negative_prompt
        self.cpu_offload, self.local_files_only = cpu_offload, local_files_only
        if pipeline_family not in {"sd1", "auto"}:
            raise ValueError("pipeline_family must be sd1 or auto.")
        if scheduler not in {
            None,
            "euler",
            "euler_ancestral",
            "dpmpp",
            "dpmpp_2m_karras",
        }:
            raise ValueError("Unsupported scheduler preset.")
        self.pipeline_family = pipeline_family
        self.scheduler_name = scheduler
        self.timestep_spacing = timestep_spacing
        self.unet_override = unet_override
        self._pipe: Any = None
        self._mode: str | None = None

    @property
    def is_mock(self) -> bool:
        """Only the explicit mock_sd identifier selects test fixtures."""
        return self.model_name == "mock_sd"

    def close(self) -> None:
        """Release the pipeline between contiguous mode batches."""
        self._pipe, self._mode = None, None
        gc.collect()
        if self.target_device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_pipeline(self, mode: str) -> Any:
        """Load native Diffusers classes; propagate download/inference errors."""
        if self._mode == mode and self._pipe is not None:
            return self._pipe
        self.close()
        from diffusers import (
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionPipeline,
        )

        classes = {
            "text2img": StableDiffusionPipeline,
            "img2img": StableDiffusionImg2ImgPipeline,
            "inpaint": StableDiffusionInpaintPipeline,
        }
        if self.pipeline_family == "auto":
            from diffusers import (
                AutoPipelineForImage2Image,
                AutoPipelineForInpainting,
                AutoPipelineForText2Image,
            )

            classes = {
                "text2img": AutoPipelineForText2Image,
                "img2img": AutoPipelineForImage2Image,
                "inpaint": AutoPipelineForInpainting,
            }
        model = self.inpaint_model_name if mode == "inpaint" else self.model_name
        revision = self.inpaint_revision if mode == "inpaint" else self.revision
        dtype = torch.float16 if self.target_device.type == "cuda" else torch.float32
        logger.info("Loading %s from %s (revision=%s)", mode, model, revision)
        loading_options: dict[str, Any] = {}
        if self.unet_override:
            from diffusers import UNet2DConditionModel
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            unet = UNet2DConditionModel.from_config(
                model, subfolder="unet", revision=revision
            )
            checkpoint = hf_hub_download(
                self.unet_override["repo"],
                self.unet_override["filename"],
                revision=self.unet_override.get("revision"),
            )
            unet.load_state_dict(load_file(checkpoint, device="cpu"), strict=True)
            loading_options["unet"] = unet.to(dtype=dtype)
        pipe = classes[mode].from_pretrained(
            model,
            revision=revision,
            torch_dtype=dtype,
            local_files_only=self.local_files_only,
            **loading_options,
        )
        if self.scheduler_name:
            from diffusers import (
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
            )

            scheduler_classes = {
                "euler": EulerDiscreteScheduler,
                "euler_ancestral": EulerAncestralDiscreteScheduler,
                "dpmpp_2m_karras": DPMSolverMultistepScheduler,
                "dpmpp": DPMSolverMultistepScheduler,
            }
            scheduler_options: dict[str, Any] = {}
            if self.timestep_spacing:
                scheduler_options["timestep_spacing"] = self.timestep_spacing
            if self.scheduler_name == "dpmpp_2m_karras":
                scheduler_options.update(
                    algorithm_type="dpmsolver++", use_karras_sigmas=True
                )
            pipe.scheduler = scheduler_classes[self.scheduler_name].from_config(
                pipe.scheduler.config, **scheduler_options
            )
        for name in (
            "unet",
            "transformer",
            "vae",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
        ):
            component = getattr(pipe, name, None)
            if component is not None:
                component.eval()
        if callable(getattr(pipe, "enable_attention_slicing", None)):
            pipe.enable_attention_slicing()
        if self.cpu_offload and self.target_device.type == "cuda":
            pipe.enable_model_cpu_offload(gpu_id=self.target_device.index or 0)
        else:
            pipe.to(self.target_device)
        self._pipe, self._mode = pipe, mode
        return pipe

    def _mock_text2img(
        self, prompt: str, height: int, width: int, seed: int
    ) -> torch.Tensor:
        """Explicit test fixture: CPU float32 (3,H,W), not a face dataset."""
        digest = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        gen = torch.Generator().manual_seed((seed + digest) % (2**63 - 1))
        return torch.rand((3, height, width), generator=gen)

    @torch.inference_mode()
    def _infer(self, mode: str, prompt: str, seed: int, **kwargs: Any) -> torch.Tensor:
        """Denoise and return finite CPU float32 RGB (3,H,W) in [0,1]."""
        pipe = self._load_pipeline(mode)
        result = pipe(
            prompt=prompt,
            negative_prompt=self.negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            **kwargs,
        )
        flags = getattr(result, "nsfw_content_detected", None)
        if flags is not None and any(flags):
            raise RuntimeError(
                "Safety checker rejected sample; no black image is saved."
            )
        array = np.asarray(result.images[0].convert("RGB"), dtype=np.float32) / 255.0
        output = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
        validate_rgb(output)
        return output

    def generate_text2img(
        self,
        prompt: str = "a photorealistic face",
        height: int = 512,
        width: int = 512,
        seed: int = 42,
    ) -> torch.Tensor:
        """Generate CPU RGB float32 (3,H,W) from text, without source identity."""
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError("Height and width must be positive multiples of eight.")
        if self.is_mock:
            return self._mock_text2img(prompt, height, width, seed)
        return self._infer("text2img", prompt, seed, height=height, width=width)

    def generate_img2img(
        self,
        source_image: torch.Tensor,
        prompt: str = "a realistic portrait",
        strength: float = 0.6,
        seed: int = 42,
    ) -> torch.Tensor:
        """Denoise RGB (3,H,W) [0,1]; real mode never blends unrelated pixels."""
        validate_rgb(source_image)
        if not 0 < strength <= 1 or int(self.num_inference_steps * strength) < 1:
            raise ValueError(
                "strength must be in (0,1] with at least one denoising step."
            )
        if self.is_mock:
            noise = self._mock_text2img(prompt, *source_image.shape[1:], seed)
            return (
                (1 - strength) * source_image.detach().cpu() + strength * noise
            ).float()
        return self._infer(
            "img2img",
            prompt,
            seed,
            image=tensor_to_pil(source_image),
            strength=strength,
        )

    def generate_inpaint(
        self,
        source_image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str = "natural smiling mouth",
        seed: int = 42,
    ) -> torch.Tensor:
        """Source RGB (3,H,W), mask (1,H,W) [0,1], 1=edit; return CPU RGB.

        Native inpainting conditions denoising on both source and mask. A final
        composite preserves source pixels outside the edit mask exactly.
        """
        validate_rgb(source_image)
        if mask.shape != (1, *source_image.shape[1:]):
            raise ValueError("Mask must be (1,H,W) matching source.")
        if (
            not torch.isfinite(mask).all()
            or mask.min() < 0
            or mask.max() > 1
            or not mask.any()
        ):
            raise ValueError("Mask must be nonempty, finite and in [0,1].")
        if self.is_mock:
            generated = self._mock_text2img(prompt, *source_image.shape[1:], seed)
        else:
            mask_image = Image.fromarray(
                np.rint(mask.detach().cpu().squeeze(0).numpy() * 255).astype(np.uint8)
            )
            generated = self._infer(
                "inpaint",
                prompt,
                seed,
                image=tensor_to_pil(source_image),
                mask_image=mask_image,
                height=source_image.shape[1],
                width=source_image.shape[2],
            )
        if generated.shape != source_image.shape:
            raise ValueError(
                "Inpainting returned wrong size; use dimensions divisible by 8."
            )
        alpha = mask.detach().cpu().float()
        return alpha * generated + (1 - alpha) * source_image.detach().cpu().float()
