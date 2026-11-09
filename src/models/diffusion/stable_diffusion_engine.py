"""Production Stable Diffusion engine supporting Text2Img, Inpainting, and Img2Img generation."""

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import zlib
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.common.device import get_optimal_device, get_torch_dtype, optimize_pipeline_memory
from src.common.logger import get_logger

logger = get_logger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, "
    "extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), "
    "disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, "
    "plastic, doll, 3d render, cartoon, anime, artificial, mannequin, fake looking, "
    "oversaturated, painting, drawing, sketch, low quality, artifacts"
)


NATIVE_SD_SIZE = 512


def generate_dynamic_face_mask(
    image: torch.Tensor,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    blur_radius: int = 16,
    target_region: str = "full",
) -> torch.Tensor:
    """Generate a smooth anatomical oval mask encompassing face, cheeks, and full jawline,
    or localized micro-regions (periocular or lips) for adversarial synthesis.

    Args:
        image: Source face tensor of shape (C, H, W) in [0, 1].
        bbox: Optional normalized (ymin, xmin, ymax, xmax) bounding box.
        blur_radius: Gaussian blur radius for boundary feathering.
        target_region: "full" for complete face/jaw, "periocular" for eye zone, "lips" for mouth zone.

    Returns:
        Feathered mask tensor of shape (1, H, W) in [0, 1].
    """
    _, h, w = image.shape
    if bbox is None:
        np_img = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            bbox = (float(fy / h), float(fx / w), float((fy + fh) / h), float((fx + fw) / w))
        else:
            bbox = (0.18, 0.22, 0.75, 0.78)

    ymin, xmin, ymax, xmax = bbox
    face_h = ymax - ymin
    face_w = xmax - xmin

    if target_region == "periocular":
        # Target periocular / eye strip (upper-mid face)
        m_ymin = ymin + 0.15 * face_h
        m_ymax = ymin + 0.58 * face_h
        m_xmin = max(0.0, xmin - 0.05 * face_w)
        m_xmax = min(1.0, xmax + 0.05 * face_w)
    elif target_region == "lips":
        # Target mouth / lips region (lower face)
        m_ymin = ymin + 0.55 * face_h
        m_ymax = min(1.0, ymax + 0.18 * face_h)
        m_xmin = xmin + 0.10 * face_w
        m_xmax = xmax - 0.10 * face_w
    else:
        # Full face: expand downwards by 18% of face height to fully cover mouth, jaw, and chin
        m_ymin = ymin
        m_ymax = min(1.0, ymax + 0.18 * face_h)
        m_xmin = max(0.0, xmin - 0.05 * face_w)
        m_xmax = min(1.0, xmax + 0.05 * face_w)

    center_y = int(((m_ymin + m_ymax) / 2.0) * h)
    center_x = int(((m_xmin + m_xmax) / 2.0) * w)
    axis_y = max(1, int(((m_ymax - m_ymin) / 2.0) * h))
    axis_x = max(1, int(((m_xmax - m_xmin) / 2.0) * w))

    mask_np = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask_np,
        (center_x, center_y),
        (axis_x, axis_y),
        angle=0,
        startAngle=0,
        endAngle=360,
        color=255,
        thickness=-1,
    )

    ksize = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=blur_radius / 2.0)
    norm_mask = blurred.astype(np.float32) / 255.0
    return torch.from_numpy(norm_mask).unsqueeze(0).to(image.device)


class StableDiffusionEngine(nn.Module):
    """Modular generation engine utilizing Stable Diffusion for deepfake image synthesis."""

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        inpaint_model_name: Optional[str] = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_mock_fallback: bool = True,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.inpaint_model_name = inpaint_model_name
        self.target_device = (
            torch.device(device) if device is not None else get_optimal_device()
        )
        self.torch_dtype = (
            torch_dtype if torch_dtype is not None else get_torch_dtype(self.target_device)
        )
        self.use_mock_fallback = use_mock_fallback
        self.default_negative_prompt = default_negative_prompt

        self._active_mode: Optional[str] = None
        self._active_pipeline: Optional[Any] = None
        logger.info(
            "Initialized StableDiffusionEngine (model=%s, inpaint_model=%s, device=%s, dtype=%s, fallback=%s)",
            self.model_name,
            self.inpaint_model_name,
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

    def _switch_mode(self, mode: str) -> Any:
        """Switch active pipeline, strictly freeing old pipeline to avoid VRAM/RAM OOM."""
        if self._active_mode == mode and self._active_pipeline is not None:
            return self._active_pipeline

        if self._active_pipeline is not None:
            logger.info("Unloading pipeline %s to free memory...", self._active_mode)
            del self._active_pipeline
            self._active_pipeline = None
            self._active_mode = None
            gc.collect()
            if self.target_device.type == "cuda":
                torch.cuda.empty_cache()

        if mode == "text2img":
            logger.info("Loading text2img pipeline from %s on %s...", self.model_name, self.target_device)
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.target_device)
        elif mode == "inpaint":
            from diffusers import StableDiffusionInpaintPipeline

            target_inpaint_model = self.inpaint_model_name or self.model_name
            logger.info("Loading inpaint pipeline from %s on %s...", target_inpaint_model, self.target_device)
            try:
                pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    target_inpaint_model,
                    torch_dtype=self.torch_dtype,
                    safety_checker=None,
                ).to(self.target_device)
            except Exception as exc:
                if target_inpaint_model != self.model_name:
                    logger.warning(
                        "Could not load specialized inpaint checkpoint '%s' (%s). Falling back to base model '%s'.",
                        target_inpaint_model,
                        exc,
                        self.model_name,
                    )
                    pipe = StableDiffusionInpaintPipeline.from_pretrained(
                        self.model_name,
                        torch_dtype=self.torch_dtype,
                        safety_checker=None,
                    ).to(self.target_device)
                else:
                    raise
        elif mode == "img2img":
            logger.info("Loading img2img pipeline from %s on %s...", self.model_name, self.target_device)
            from diffusers import StableDiffusionImg2ImgPipeline

            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                safety_checker=None,
            ).to(self.target_device)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        optimize_pipeline_memory(pipe)
        self._active_pipeline = pipe
        self._active_mode = mode
        return self._active_pipeline

    def get_text2img_pipeline(self) -> Any:
        """Get or initialize StableDiffusionPipeline."""
        return self._switch_mode("text2img")

    def get_inpaint_pipeline(self) -> Any:
        """Get or initialize StableDiffusionInpaintPipeline."""
        return self._switch_mode("inpaint")

    def get_img2img_pipeline(self) -> Any:
        """Get or initialize StableDiffusionImg2ImgPipeline."""
        return self._switch_mode("img2img")

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
        if self.target_device.type == "cuda":
            torch.cuda.empty_cache()
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
        if self.target_device.type == "cuda":
            torch.cuda.empty_cache()
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
        if self.target_device.type == "cuda":
            torch.cuda.empty_cache()
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
