"""Stable Diffusion engine supporting Text2Img, Inpainting, and Img2Img generation."""

from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn

from src.common.logger import get_logger

logger = get_logger(__name__)


class StableDiffusionEngine(nn.Module):
    """Modular generation engine utilizing Stable Diffusion for deepfake image synthesis."""

    def __init__(
        self,
        model_name: str = "mock_sd",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Initialized StableDiffusionEngine (model={self.model_name}, device={self.target_device})")

    def _mock_text2img(self, prompt: str, height: int, width: int, seed: int = 42) -> torch.Tensor:
        """Deterministic mock text-to-image synthesis producing valid portrait patterns."""
        gen = torch.Generator().manual_seed(seed + hash(prompt) % 10000)
        # Create structured base portrait pattern
        y = torch.linspace(-1.0, 1.0, height).view(height, 1).repeat(1, width)
        x = torch.linspace(-1.0, 1.0, width).view(1, width).repeat(height, 1)
        r = torch.sqrt(x**2 + y**2) # Radial facial oval

        face_mask = (r < 0.8).float()
        noise = torch.rand(3, height, width, generator=gen) * 0.2
        pattern = torch.zeros(3, height, width)
        pattern[0] = 0.75 * face_mask + 0.1 # Skin tone channel
        pattern[1] = 0.55 * face_mask + 0.1
        pattern[2] = 0.45 * face_mask + 0.1

        img = torch.clamp(pattern + noise, 0.0, 1.0)
        return img.to(self.target_device)

    def generate_text2img(
        self,
        prompt: str = "a photorealistic face",
        height: int = 256,
        width: int = 256,
        seed: int = 42,
    ) -> torch.Tensor:
        """Synthesize image from scratch conditioned on text prompt.
        
        Args:
            prompt: Text prompt description.
            height: Image height H.
            width: Image width W.
            seed: Random seed.
            
        Returns:
            torch.Tensor of shape (3, H, W) normalized to [0.0, 1.0].
        """
        if self.model_name.startswith("mock"):
            return self._mock_text2img(prompt, height, width, seed=seed)

        try:
            from diffusers import StableDiffusionPipeline
            # Real GPU Diffusers inference
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            ).to(self.target_device)
            gen = torch.Generator(device=self.target_device).manual_seed(seed)
            res = pipe(prompt=prompt, height=height, width=width, generator=gen).images[0]
            np_img = np.array(res, dtype=np.float32) / 255.0
            return torch.from_numpy(np_img).permute(2, 0, 1).contiguous()
        except Exception as e:
            logger.warning(f"Diffusers execution failed ({e}), falling back to mock mode.")
            return self._mock_text2img(prompt, height, width, seed=seed)

    def generate_inpaint(
        self,
        source_image: torch.Tensor,
        mask: torch.Tensor,
        prompt: str = "realistic facial feature",
        seed: int = 42,
    ) -> torch.Tensor:
        """Modify specific facial region defined by mask while strictly preserving background.
        
        Args:
            source_image: Tensor of shape (3, H, W) in [0.0, 1.0].
            mask: Binary tensor of shape (1, H, W) where 1 indicates region to inpaint.
            prompt: Text prompt conditioning for the masked region.
            seed: Random seed.
            
        Returns:
            Inpainted tensor of shape (3, H, W) in [0.0, 1.0].
        """
        c, h, w = source_image.shape
        gen_img = self.generate_text2img(prompt=prompt, height=h, width=w, seed=seed)

        # Inpainting composite contract: M * I_gen + (1 - M) * I_source
        mask_3c = mask.expand(3, h, w).to(source_image.device)
        inpainted = mask_3c * gen_img.to(source_image.device) + (1.0 - mask_3c) * source_image
        return torch.clamp(inpainted, 0.0, 1.0)

    def generate_img2img(
        self,
        source_image: torch.Tensor,
        prompt: str = "photorealistic face alteration",
        strength: float = 0.6,
        seed: int = 42,
    ) -> torch.Tensor:
        """Alter existing facial image conditioned on prompt and strength.
        
        Args:
            source_image: Tensor of shape (3, H, W) in [0.0, 1.0].
            prompt: Text prompt describing alteration.
            strength: Degree of alteration in range [0.0, 1.0].
            seed: Random seed.
            
        Returns:
            Altered tensor of shape (3, H, W) in [0.0, 1.0].
        """
        c, h, w = source_image.shape
        gen_img = self.generate_text2img(prompt=prompt, height=h, width=w, seed=seed)

        # Img2Img blend contract: (1 - strength) * source + strength * gen
        blend = (1.0 - strength) * source_image + strength * gen_img.to(source_image.device)
        return torch.clamp(blend, 0.0, 1.0)
