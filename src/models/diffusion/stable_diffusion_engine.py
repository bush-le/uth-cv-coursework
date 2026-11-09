"""Production Stable Diffusion engine supporting Text2Img, Inpainting, and Img2Img generation."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import zlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.common.device import get_optimal_device, get_torch_dtype, optimize_pipeline_memory
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "deformed, blurry, bad anatomy, bad eyes, disfigured, poorly drawn face, "
    "mutation, mutated, extra_limbs, low quality, artifacts"
)


NATIVE_SD_SIZE = 512


class StableDiffusionEngine(nn.Module):
    """Modular generation engine utilizing Stable Diffusion for deepfake image synthesis."""

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_mock_fallback: bool = True,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.target_device = (
            torch.device(device) if device is not None else get_optimal_device()
        )
        self.torch_dtype = (
            torch_dtype if torch_dtype is not None else get_torch_dtype(self.target_device)
        )
        self.use_mock_fallback = use_mock_fallback
        self.default_negative_prompt = default_negative_prompt

        self._pipelines: Dict[str, Any] = {}
        logger.info(
            "Initialized StableDiffusionEngine (model=%s, device=%s, dtype=%s, fallback=%s)",
            self.model_name,
            self.target_device,
            self.torch_dtype,
            self.use_mock_fallback,
        )

    def _mock_text2img(self, prompt: str, height: int, width: int, seed: int = 42) -> torch.Tensor:
        """Deterministic mock text-to-image synthesis producing structured portrait patterns."""
        seed_offset = zlib.crc32(prompt.encode("utf-8")) % 10000
        gen = torch.Generator().manual_seed((seed + seed_offset) % (2**31 - 1))
        y = torch.linspace(-1.0, 1.0, height).view(height, 1).repeat(1, width)
        x = torch.linspace(-1.0, 1.0, width).view(1, width).repeat(height, 1)
        r = torch.sqrt(x**2 + y**2)

        face_mask = (r < 0.8).float()
        noise = torch.rand(3, height, width, generator=gen) * 0.2
        pattern = torch.zeros(3, height, width)
        pattern[0] = 0.75 * face_mask + 0.1
        pattern[1] = 0.55 * face_mask + 0.1
        pattern[2] = 0.45 * face_mask + 0.1

        img = torch.clamp(pattern + noise, 0.0, 1.0)
        return img.to(self.target_device)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert float tensor (3, H, W) in [0.0, 1.0] to RGB PIL Image."""
        tensor_cpu = tensor.detach().cpu().clamp(0.0, 1.0)
        np_arr = (tensor_cpu.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_arr)

    def _mask_to_pil(self, mask: torch.Tensor) -> Image.Image:
        """Convert binary mask tensor (1, H, W) or (H, W) in [0.0, 1.0] to L PIL Image."""
        mask_cpu = mask.detach().cpu().clamp(0.0, 1.0)
        if mask_cpu.ndim == 3:
            mask_cpu = mask_cpu.squeeze(0)
        np_mask = (mask_cpu.numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_mask, mode="L")

    def _pil_to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert PIL Image to float RGB tensor (3, H, W) in [0.0, 1.0]."""
        rgb_img = pil_img.convert("RGB")
        np_arr = np.array(rgb_img, dtype=np.float32) / 255.0
        return torch.from_numpy(np_arr).permute(2, 0, 1).contiguous()

    def get_text2img_pipeline(self) -> Any:
        """Get or initialize StableDiffusionPipeline."""
        if "text2img" not in self._pipelines:
            from diffusers import StableDiffusionPipeline

            logger.info("Loading text2img pipeline from %s...", self.model_name)
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.target_device)
            optimize_pipeline_memory(pipe)
            self._pipelines["text2img"] = pipe
        return self._pipelines["text2img"]

    def get_inpaint_pipeline(self) -> Any:
        """Get or initialize StableDiffusionInpaintPipeline."""
        if "inpaint" not in self._pipelines:
            from diffusers import StableDiffusionInpaintPipeline

            logger.info("Loading inpainting pipeline from %s...", self.model_name)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.target_device)
            optimize_pipeline_memory(pipe)
            self._pipelines["inpaint"] = pipe
        return self._pipelines["inpaint"]

    def get_img2img_pipeline(self) -> Any:
        """Get or initialize StableDiffusionImg2ImgPipeline."""
        if "img2img" not in self._pipelines:
            from diffusers import StableDiffusionImg2ImgPipeline

            logger.info("Loading img2img pipeline from %s...", self.model_name)
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.target_device)
            optimize_pipeline_memory(pipe)
            self._pipelines["img2img"] = pipe
        return self._pipelines["img2img"]

    def generate_text2img(
        self,
        prompt: str = "a photorealistic face",
        height: int = 256,
        width: int = 256,
        seed: int = 42,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Synthesize image from scratch conditioned on text prompt."""
        if self.model_name.startswith("mock"):
            return self._mock_text2img(prompt, height, width, seed=seed)

        neg = negative_prompt if negative_prompt is not None else self.default_negative_prompt
        try:
            pipe = self.get_text2img_pipeline()
            gen = torch.Generator(device=self.target_device).manual_seed(seed)
            diff_h = max(NATIVE_SD_SIZE, height)
            diff_w = max(NATIVE_SD_SIZE, width)
            out_img = pipe(
                prompt=prompt,
                negative_prompt=neg,
                height=diff_h,
                width=diff_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            ).images[0]
            if (out_img.width, out_img.height) != (width, height):
                out_img = out_img.resize((width, height), Image.Resampling.LANCZOS)
            return self._pil_to_tensor(out_img).to(self.target_device)
        except Exception as e:
            if self.use_mock_fallback:
                logger.warning("Diffusers text2img failed (%s), using mock fallback.", e)
                return self._mock_text2img(prompt, height, width, seed=seed)
            raise e

    def generate_inpaint(
        self,
        source_image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str = "realistic facial feature",
        seed: int = 42,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Modify specific facial region defined by mask while strictly preserving background."""
        c, h, w = source_image.shape
        if self.model_name.startswith("mock"):
            gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
            mask_3c = mask.expand(3, h, w).to(source_image.device)
            inpainted = mask_3c * gen_img.to(source_image.device) + (1.0 - mask_3c) * source_image
            return torch.clamp(inpainted, 0.0, 1.0)

        neg = negative_prompt if negative_prompt is not None else self.default_negative_prompt
        try:
            from PIL import ImageFilter

            pipe = self.get_inpaint_pipeline()
            diff_h = max(NATIVE_SD_SIZE, h)
            diff_w = max(NATIVE_SD_SIZE, w)
            pil_source = self._tensor_to_pil(source_image).resize((diff_w, diff_h), Image.Resampling.LANCZOS)
            pil_mask = self._mask_to_pil(mask).resize((diff_w, diff_h), Image.Resampling.BILINEAR)
            # Soften mask boundary to prevent high-frequency aliasing and barcode artifacts in 4-channel UNet
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=8))

            gen = torch.Generator(device=self.target_device).manual_seed(seed)
            out_img = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=pil_source,
                mask_image=pil_mask,
                height=diff_h,
                width=diff_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            ).images[0]
            if (out_img.width, out_img.height) != (w, h):
                out_img = out_img.resize((w, h), Image.Resampling.LANCZOS)
            return self._pil_to_tensor(out_img).to(self.target_device)
        except Exception as e:
            if self.use_mock_fallback:
                logger.warning("Diffusers inpaint failed (%s), using mock fallback.", e)
                gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
                mask_3c = mask.expand(3, h, w).to(source_image.device)
                inpainted = mask_3c * gen_img.to(source_image.device) + (1.0 - mask_3c) * source_image
                return torch.clamp(inpainted, 0.0, 1.0)
            raise e

    def generate_img2img(
        self,
        source_image: torch.Tensor,
        prompt: str = "photorealistic face alteration",
        strength: float = 0.40,
        seed: int = 42,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Alter existing facial image conditioned on prompt and strength."""
        c, h, w = source_image.shape
        if self.model_name.startswith("mock"):
            gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
            blend = (1.0 - strength) * source_image + strength * gen_img.to(source_image.device)
            return torch.clamp(blend, 0.0, 1.0)

        neg = negative_prompt if negative_prompt is not None else self.default_negative_prompt
        try:
            pipe = self.get_img2img_pipeline()
            diff_h = max(NATIVE_SD_SIZE, h)
            diff_w = max(NATIVE_SD_SIZE, w)
            pil_source = self._tensor_to_pil(source_image).resize((diff_w, diff_h), Image.Resampling.LANCZOS)
            gen = torch.Generator(device=self.target_device).manual_seed(seed)
            out_img = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=pil_source,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=gen,
            ).images[0]
            if (out_img.width, out_img.height) != (w, h):
                out_img = out_img.resize((w, h), Image.Resampling.LANCZOS)
            return self._pil_to_tensor(out_img).to(self.target_device)
        except Exception as e:
            if self.use_mock_fallback:
                logger.warning("Diffusers img2img failed (%s), using mock fallback.", e)
                gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
                blend = (1.0 - strength) * source_image + strength * gen_img.to(source_image.device)
                return torch.clamp(blend, 0.0, 1.0)
            raise e
