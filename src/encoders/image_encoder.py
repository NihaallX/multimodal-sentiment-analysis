"""
CGRN — Image Encoder
=====================
Wraps MobileNetV3-Small (default) or EfficientNet-Lite to produce a
fixed-size, L2-normalized sentiment embedding vector.

Architecture
------------
  MobileNetV3-Small / EfficientNet
       ↓  global average-pooled features  (576 or 1280 dim)
  Linear projection  →  embed_dim  (default 256)
  LayerNorm
  L2 Normalization
       ↓
  S_i  ∈ R^{embed_dim}   (unit-normalized sentiment vector)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision import transforms


class ImageEncoder(nn.Module):
    """
    Parameters
    ----------
    backbone_name : str
        One of: 'mobilenet_v3_small', 'mobilenet_v3_large',
                'efficientnet_b0', 'efficientnet_b1'
    embed_dim : int
        Output sentiment embedding dimension (default 256).
    freeze_backbone : bool
        If True, backbone weights are frozen; only projection head trains.
    dropout : float
        Dropout applied before projection.
    pretrained : bool
        Load ImageNet pretrained weights.
    """

    BACKBONE_FEATURE_DIM = {
        "mobilenet_v3_small": 576,
        "mobilenet_v3_large": 960,
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
    }

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_small",
        embed_dim: int = 256,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embed_dim = embed_dim

        weights_arg = "DEFAULT" if pretrained else None

        # --- Backbone ---------------------------------------------------------
        if backbone_name == "mobilenet_v3_small":
            base = tvm.mobilenet_v3_small(weights=weights_arg)
            feature_dim = self.BACKBONE_FEATURE_DIM["mobilenet_v3_small"]
            # Remove final classifier, keep feature extractor
            self.backbone = nn.Sequential(*list(base.children())[:-1])   # up to avgpool

        elif backbone_name == "mobilenet_v3_large":
            base = tvm.mobilenet_v3_large(weights=weights_arg)
            feature_dim = self.BACKBONE_FEATURE_DIM["mobilenet_v3_large"]
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        elif backbone_name == "efficientnet_b0":
            base = tvm.efficientnet_b0(weights=weights_arg)
            feature_dim = self.BACKBONE_FEATURE_DIM["efficientnet_b0"]
            self.backbone = nn.Sequential(base.features, base.avgpool)

        elif backbone_name == "efficientnet_b1":
            base = tvm.efficientnet_b1(weights=weights_arg)
            feature_dim = self.BACKBONE_FEATURE_DIM["efficientnet_b1"]
            self.backbone = nn.Sequential(base.features, base.avgpool)

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. "
                             f"Choose from {list(self.BACKBONE_FEATURE_DIM.keys())}")

        self.feature_dim = feature_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # --- Projection head --------------------------------------------------
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(feature_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # --- Unimodal classification head (used in Stage 1) ------------------
        self.classifier = nn.Linear(embed_dim, 3)   # 3-class: neg/neu/pos

        self._init_weights()

    # -------------------------------------------------------------------------
    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    # -------------------------------------------------------------------------
    @staticmethod
    def get_transforms(split: str = "train", image_size: int = 224):
        """
        Returns standard torchvision transforms for train / val / test.
        """
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        if split == "train":
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:   # val / test
            return transforms.Compose([
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    # -------------------------------------------------------------------------
    def encode(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Parameters
        ----------
        images : Tensor [B, 3, H, W]

        Returns
        -------
        embedding : Tensor [B, embed_dim]
            L2-normalized sentiment vector if normalize=True.
        """
        features = self.backbone(images)          # [B, C, 1, 1] or [B, C, 1, 1]
        features = self.flatten(features)         # [B, feature_dim]
        x = self.dropout(features)
        x = self.proj(x)                          # [B, embed_dim]
        x = self.norm(x)
        if normalize:
            x = F.normalize(x, p=2, dim=-1)
        return x

    # -------------------------------------------------------------------------
    def forward(self, images: torch.Tensor, normalize: bool = True):
        """
        Returns
        -------
        embedding : Tensor [B, embed_dim]   — sentiment vector S_i
        logits    : Tensor [B, 3]           — unimodal classification logits
        """
        emb = self.encode(images, normalize=normalize)
        logits = self.classifier(emb)
        return emb, logits

    # -------------------------------------------------------------------------
    def param_count(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone  = sum(p.numel() for p in self.backbone.parameters())
        head      = sum(p.numel() for p in self.proj.parameters()) + \
                    sum(p.numel() for p in self.norm.parameters()) + \
                    sum(p.numel() for p in self.classifier.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "projection_head": head,
        }


# =============================================================================
# Convenience factory
# =============================================================================

def build_image_encoder(
    backbone_name: str = "mobilenet_v3_small",
    embed_dim: int = 256,
    freeze_backbone: bool = False,
) -> ImageEncoder:
    """Factory function for ImageEncoder."""
    return ImageEncoder(
        backbone_name=backbone_name,
        embed_dim=embed_dim,
        freeze_backbone=freeze_backbone,
    )
