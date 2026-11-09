"""DINO ViT (dinoViT) and CNN deepfake image detector architectures for baseline benchmarking, hard-sample mining, and fine-tuning."""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from src.common.logger import get_logger

logger = get_logger(__name__)


class DinoViTBackbone(nn.Module):
    """Self-supervised Vision Transformer (DINO ViT) feature extractor with dynamic positional interpolation.

    Extracts global [CLS] token and patch representations from 2D facial images of arbitrary resolution.

    Tensor Input Contract:
        x: (B, 3, H, W) float32 RGB image tensor.
    Tensor Output Contract:
        cls_embedding: (B, embed_dim) normalized feature vector.
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        base_grid_size: int = 16,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.base_grid_size = base_grid_size

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, base_grid_size * base_grid_size, embed_dim) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _interpolate_pos_encoding(self, h_grid: int, w_grid: int) -> torch.Tensor:
        """Interpolate 2D positional embeddings to match input spatial patch grid (h_grid, w_grid)."""
        if h_grid == self.base_grid_size and w_grid == self.base_grid_size:
            return self.pos_embed

        pos_2d = self.pos_embed.reshape(
            1, self.base_grid_size, self.base_grid_size, self.embed_dim
        ).permute(0, 3, 1, 2)
        pos_interp = F.interpolate(
            pos_2d, size=(h_grid, w_grid), mode="bilinear", align_corners=False
        )
        return pos_interp.permute(0, 2, 3, 1).reshape(1, h_grid * w_grid, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute DINO ViT [CLS] token representation.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Normalized [CLS] token embedding of shape (B, embed_dim).
        """
        b, c, h, w = x.shape
        if h < self.patch_size or w < self.patch_size:
            x = F.interpolate(
                x,
                size=(max(h, self.patch_size), max(w, self.patch_size)),
                mode="bilinear",
                align_corners=False,
            )

        patches = self.patch_embed(x)  # (B, embed_dim, H_grid, W_grid)
        h_grid, w_grid = patches.shape[2], patches.shape[3]
        tokens = patches.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        tokens = tokens + self._interpolate_pos_encoding(h_grid, w_grid)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        seq = torch.cat((cls_tokens, tokens), dim=1)  # (B, 1 + N_patches, embed_dim)

        for block in self.blocks:
            seq = block(seq)

        seq = self.norm(seq)
        cls_out = seq[:, 0]  # (B, embed_dim)
        return F.normalize(cls_out, p=2, dim=-1)


class MetaDinoV2Wrapper(nn.Module):
    """Production Meta DINOv2 feature extractor leveraging pretrained Vision Transformer."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.model_name = model_name
        self.pretrained = pretrained
        try:
            if pretrained:
                self.backbone = AutoModel.from_pretrained(model_name)
            else:
                cfg = AutoConfig.from_pretrained(model_name)
                self.backbone = AutoModel.from_config(cfg)
        except Exception as e:
            logger.warning("Failed to load %s from Hub (%s), using local config.", model_name, e)
            cfg = AutoConfig.from_pretrained(model_name)
            self.backbone = AutoModel.from_config(cfg)

        self.embed_dim = self.backbone.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract normalized [CLS] token embedding of shape (B, embed_dim)."""
        b, c, h, w = x.shape
        if h % 14 != 0 or w % 14 != 0:
            h_new = max(14, (h // 14) * 14)
            w_new = max(14, (w // 14) * 14)
            x = F.interpolate(x, size=(h_new, w_new), mode="bilinear", align_corners=False)

        outputs = self.backbone(pixel_values=x)
        cls_out = outputs.last_hidden_state[:, 0]
        return F.normalize(cls_out, p=2, dim=-1)


class SimpleImageCNN(nn.Module):
    """Lightweight 2D convolutional feature extractor for ablation comparisons."""

    def __init__(self, out_features: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class DeepfakeImageDetector(nn.Module):
    """Binary classifier detecting synthetic vs real facial images using DINO ViT (dinoViT) baseline.

    Supports both frozen baseline evaluation (to mine bypassed hard samples) and
    end-to-end / partial-block fine-tuning on Baseline + Synthetic + Hard Sample datasets.

    Tensor Input Contract:
        x: (B, 3, H, W)
    Tensor Output Contract:
        logits: (B, num_classes)
    """

    def __init__(
        self,
        backbone: str = "dino_vit",
        num_classes: int = 1,
        dropout: float = 0.3,
        freeze_extractor: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone.lower()
        self.num_classes = num_classes

        if self.backbone_name in ("dinov2_small", "dinov2", "dinov2_vits14", "facebook/dinov2-small"):
            self.extractor: nn.Module = MetaDinoV2Wrapper(
                model_name="facebook/dinov2-small",
                pretrained=pretrained,
            )
            feature_dim = self.extractor.embed_dim
        elif self.backbone_name in ("dino_vit", "dinovit", "dino_vit_tiny"):
            feature_dim = 192
            self.extractor = DinoViTBackbone(
                patch_size=16,
                embed_dim=feature_dim,
                depth=4,
                num_heads=4,
            )
        elif self.backbone_name == "resnet18":
            base = resnet18(weights=None)
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()  # type: ignore
            self.extractor = base
        else:
            feature_dim = 128
            self.extractor = SimpleImageCNN(out_features=feature_dim)

        self.feature_dim = feature_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes),
        )

        if freeze_extractor:
            self.freeze_backbone()

        logger.info(
            "Initialized DeepfakeImageDetector: backbone=%s, classes=%d, frozen=%s",
            self.backbone_name,
            num_classes,
            freeze_extractor,
        )

    def freeze_backbone(self) -> None:
        """Freeze feature extractor weights for baseline linear probing / hard-sample mining."""
        for param in self.extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None) -> None:
        """Unfreeze DINO ViT transformer blocks for fine-tuning on hard samples.

        Args:
            unfreeze_last_n_blocks: If provided and extractor is DinoViTBackbone or MetaDinoV2Wrapper,
                                    only unfreezes the last N transformer blocks. Otherwise
                                    unfreezes the entire backbone.
        """
        if (
            unfreeze_last_n_blocks is not None
            and isinstance(self.extractor, MetaDinoV2Wrapper)
            and unfreeze_last_n_blocks > 0
        ):
            self.freeze_backbone()
            layers = self.extractor.backbone.encoder.layer
            for layer in list(layers)[-unfreeze_last_n_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif (
            unfreeze_last_n_blocks is not None
            and isinstance(self.extractor, DinoViTBackbone)
            and unfreeze_last_n_blocks > 0
        ):
            self.freeze_backbone()
            for block in list(self.extractor.blocks)[-unfreeze_last_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.extractor.norm.parameters():
                param.requires_grad = True
        else:
            for param in self.extractor.parameters():
                param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone embedding vector of shape (B, feature_dim)."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D image tensor (B, 3, H, W), got shape {tuple(x.shape)}")
        return self.extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for image batch.

        Args:
            x: torch.Tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor of shape (B, num_classes) representing classification logits.
        """
        feats = self.extract_features(x)
        logits = self.classifier(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sigmoid probability P(fake) in [0.0, 1.0] for binary detection."""
        return torch.sigmoid(self.forward(x))
