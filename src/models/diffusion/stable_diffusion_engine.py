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


CALIBRATED_NEGATIVE_PROMPT = (
    "blurry, distorted face, bad anatomy, deformed eyes, extra fingers, cartoon, "
    "3d render, doll, oversaturated, plastic skin, mutation, cropped forehead, unitooth, out of frame"
)
DEFAULT_NEGATIVE_PROMPT = CALIBRATED_NEGATIVE_PROMPT


NATIVE_SD_SIZE = 512



def generate_dynamic_face_mask(
    image: torch.Tensor,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    blur_radius: int = 20,
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

    # Detect if bbox is missing, pixel-coordinate formatted, or a dummy full-canvas bbox
    needs_detection = False
    if bbox is None:
        needs_detection = True
    elif any(coord > 1.0 for coord in bbox):
        # Pixel coordinates passed (e.g. [0, 0, 1024, 1024] or [x1, y1, x2, y2])
        bx1, by1, bx2, by2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        bw, bh = abs(bx2 - bx1), abs(by2 - by1)
        max_coord = max(bx1, by1, bx2, by2)
        ref_w = 1024.0 if max_coord > max(w, h) else float(w)
        ref_h = 1024.0 if max_coord > max(w, h) else float(h)
        if bw >= 0.90 * ref_w and bh >= 0.90 * ref_h:
            needs_detection = True
        else:
            bbox = (min(by1, by2) / ref_h, min(bx1, bx2) / ref_w, max(by1, by2) / ref_h, max(bx1, bx2) / ref_w)
    elif (bbox[2] - bbox[0] >= 0.90) and (bbox[3] - bbox[1] >= 0.90):
        # Normalized but spans >= 90% of frame -> raw canvas dummy box
        needs_detection = True

    if needs_detection:
        try:
            np_img = (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    bbox = (float(fy / h), float(fx / w), float((fy + fh) / h), float((fx + fw) / w))
                else:
                    bbox = (0.18, 0.22, 0.75, 0.78)
            else:
                bbox = (0.18, 0.22, 0.75, 0.78)
        except Exception:
            bbox = (0.18, 0.22, 0.75, 0.78)
        ymin, xmin, ymax, xmax = bbox
    else:
        # Standardize external Pascal VOC [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = bbox

    face_h = ymax - ymin
    face_w = xmax - xmin

    if target_region == "periocular":
        # Target periocular / eye strip (upper-mid face)
        m_ymin = max(0.0, min(1.0, ymin + 0.15 * face_h))
        m_ymax = max(0.0, min(1.0, ymin + 0.58 * face_h))
        m_xmin = max(0.0, min(1.0, xmin - 0.05 * face_w))
        m_xmax = max(0.0, min(1.0, xmax + 0.05 * face_w))
    elif target_region == "lips":
        # Target mouth / lips region (strictly lower face without neck)
        m_ymin = max(0.0, min(1.0, ymin + 0.60 * face_h))
        m_ymax = max(0.0, min(1.0, ymin + 0.88 * face_h))
        m_xmin = max(0.0, min(1.0, xmin + 0.10 * face_w))
        m_xmax = max(0.0, min(1.0, xmax - 0.10 * face_w))
    elif target_region == "nose":
        # Target central nasal bridge and tip
        m_ymin = max(0.0, min(1.0, ymin + 0.35 * face_h))
        m_ymax = max(0.0, min(1.0, ymin + 0.68 * face_h))
        m_xmin = max(0.0, min(1.0, xmin + 0.15 * face_w))
        m_xmax = max(0.0, min(1.0, xmax - 0.15 * face_w))
    else:
        # Full face: expand downwards by 18% of face height to fully cover mouth, jaw, and chin
        m_ymin = max(0.0, min(1.0, ymin))
        m_ymax = max(0.0, min(1.0, ymax + 0.18 * face_h))
        m_xmin = max(0.0, min(1.0, xmin - 0.05 * face_w))
        m_xmax = max(0.0, min(1.0, xmax + 0.05 * face_w))

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
    if norm_mask.sum() < 10.0:
        cv2.ellipse(mask_np, (w // 2, h // 2), (w // 4, h // 3), 0, 0, 360, 255, -1)
        blurred = cv2.GaussianBlur(mask_np, (ksize, ksize), sigmaX=blur_radius / 2.0)
        norm_mask = blurred.astype(np.float32) / 255.0
    return torch.from_numpy(norm_mask).unsqueeze(0).to(image.device)


class StableDiffusionEngine(nn.Module):
    """Modular generation engine utilizing Stable Diffusion for deepfake image synthesis."""

    def __init__(
        self,
        model_name: str = DEFAULT_SD,
        inpaint_model_name: Optional[str] = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        use_mock_fallback: bool = False,
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        *,
        revision: Optional[str] = None,
        inpaint_revision: Optional[str] = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        cpu_offload: bool = False,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.inpaint_model_name = inpaint_model_name or model_name
        self.target_device = (
            torch.device(device) if device is not None else get_optimal_device()
        )
        self.torch_dtype = (
            torch_dtype if torch_dtype is not None else get_torch_dtype(self.target_device)
        )
        self.use_mock_fallback = use_mock_fallback
        self.default_negative_prompt = default_negative_prompt
        self.revision = revision
        self.inpaint_revision = inpaint_revision
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt if negative_prompt is not None else default_negative_prompt
        self.cpu_offload = cpu_offload
        self.local_files_only = local_files_only

        self._active_mode: Optional[str] = None
        self._active_pipeline: Optional[Any] = None
        self._pipe: Optional[Any] = None
        self._mode: Optional[str] = None
        logger.info(
            "Initialized StableDiffusionEngine (model=%s, inpaint_model=%s, device=%s, dtype=%s, fallback=%s)",
            self.model_name,
            self.inpaint_model_name,
            self.target_device,
            self.torch_dtype,
            self.use_mock_fallback,
        )

    @property
    def is_mock(self) -> bool:
        """Only the explicit mock_sd identifier selects test fixtures."""
        return self.model_name == "mock_sd"

    def close(self) -> None:
        """Release the pipeline between contiguous mode batches."""
        self._pipe, self._mode = None, None
        self._active_pipeline, self._active_mode = None, None
        gc.collect()
        if self.target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_pipeline(self, mode: str) -> Any:
        """Load native Diffusers classes; propagate download/inference errors."""
        return self._switch_mode(mode)

    def _infer(self, mode: str, prompt: str, seed: int, **kwargs: Any) -> torch.Tensor:
        """Denoise and return finite CPU float32 RGB (3, H, W) in [0, 1]."""
        pipe = self._load_pipeline(mode)
        neg_prompt = kwargs.pop("negative_prompt", self.negative_prompt)
        steps = kwargs.pop("num_inference_steps", self.num_inference_steps)
        guidance = kwargs.pop("guidance_scale", self.guidance_scale)
        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            **kwargs,
        )
        flags = getattr(result, "nsfw_content_detected", None)
        if flags is not None and any(flags):
            raise RuntimeError(
                "Safety checker rejected generation; review prompt and inputs."
            )
        image = result.images[0]
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        validate_rgb(tensor)
        return tensor

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

        # Configure DPMSolverMultistepScheduler for high-fidelity 20-step generation (5.47s/img vs 7.38s/img)
        try:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
            logger.info("Configured DPMSolverMultistepScheduler (karras_sigmas=True, dpmsolver++) on %s", mode)
        except Exception as exc:
            logger.warning("Could not set DPMSolverMultistepScheduler (%s). Keeping default scheduler.", exc)

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

    @torch.no_grad()
    def generate_text2img(
        self,
        prompt: str = "a photorealistic face",
        height: int = 512,
        width: int = 512,
        seed: int = 42,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Synthesize image from scratch conditioned on text prompt."""
        if height < 8 or width < 8 or height % 8 or width % 8:
            raise ValueError("Height and width must be positive multiples of eight.")
        if self.model_name.startswith("mock"):
            return self._mock_text2img(prompt, height, width, seed=seed)

        neg = negative_prompt if negative_prompt is not None else self.negative_prompt
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

        try:
            return self._infer(
                "text2img",
                prompt,
                seed,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=neg,
            )
        except Exception as e:
            if self.use_mock_fallback:
                logger.warning("Diffusers text2img failed (%s), using mock fallback.", e)
                return self._mock_text2img(prompt, height, width, seed=seed)
            raise e

    @torch.no_grad()
    def generate_inpaint(
        self,
        source_image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str = "realistic facial feature",
        seed: int = 42,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Modify specific facial region defined by mask while strictly preserving background."""
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

        c, h, w = source_image.shape
        if self.model_name.startswith("mock"):
            gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
            mask_expanded = mask.expand(3, h, w).to(source_image.device)
            m = torch.clamp(mask_expanded, 0.0, 1.0)
            m_smooth = m * m * (3.0 - 2.0 * m)
            blended = m_smooth * gen_img.to(source_image.device) + (1.0 - m_smooth) * source_image
            return torch.clamp(blended, 0.0, 1.0)

        neg = negative_prompt if negative_prompt is not None else self.negative_prompt
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        guidance = guidance_scale if guidance_scale is not None else self.guidance_scale

        pil_source = tensor_to_pil(source_image)
        mask_np = np.rint(mask.detach().cpu().squeeze(0).numpy() * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_np, mode="L")
        pil_mask_binary = pil_mask.point(lambda p: 255 if p > 128 else 0)

        try:
            inpainted_tensor = self._infer(
                "inpaint",
                prompt,
                seed,
                image=pil_source,
                mask_image=pil_mask_binary,
                height=h,
                width=w,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=neg,
            ).to(source_image.device)
            mask_expanded = mask.expand_as(source_image).to(source_image.device)
            m = torch.clamp(mask_expanded, 0.0, 1.0)
            blended = m * inpainted_tensor + (1.0 - m) * source_image
            return torch.clamp(blended, 0.0, 1.0)
        except Exception as e:
            if self.use_mock_fallback:
                logger.warning("Diffusers inpaint failed (%s), using mock fallback.", e)
                gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
                mask_expanded = mask.expand(3, h, w).to(source_image.device)
                m = torch.clamp(mask_expanded, 0.0, 1.0)
                m_smooth = m * m * (3.0 - 2.0 * m)
                blended = m_smooth * gen_img.to(source_image.device) + (1.0 - m_smooth) * source_image
                return torch.clamp(blended, 0.0, 1.0)
            raise e

    @torch.no_grad()
    def generate_img2img(
        self,
        source_image: torch.Tensor,
        prompt: str = "photorealistic face alteration",
        strength: float = 0.55,
        seed: int = 42,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """Alter existing facial image conditioned on prompt and strength."""
        validate_rgb(source_image)
        steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        if not 0 < strength <= 1 or int(steps * strength) < 1:
            raise ValueError("strength must be in (0,1] with at least one denoising step.")

        c, h, w = source_image.shape
        if self.model_name.startswith("mock"):
            gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
            blend = (1.0 - strength) * source_image + strength * gen_img.to(source_image.device)
            return torch.clamp(blend, 0.0, 1.0)

        neg = negative_prompt if negative_prompt is not None else self.negative_prompt
        guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        pil_source = tensor_to_pil(source_image)
        try:
            return self._infer(
                "img2img",
                prompt,
                seed,
                image=pil_source,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=neg,
            ).to(source_image.device)
        except Exception as e:
            if self.use_mock_fallback:
                logger.warning("Diffusers img2img failed (%s), using mock fallback.", e)
                gen_img = self._mock_text2img(prompt=prompt, height=h, width=w, seed=seed)
                blend = (1.0 - strength) * source_image + strength * gen_img.to(source_image.device)
                return torch.clamp(blend, 0.0, 1.0)
            raise e
