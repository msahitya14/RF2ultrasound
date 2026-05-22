"""
model.py — EfficientNet-B0 backbone for (x, y) regression.
Uses pretrained ImageNet weights, fine-tuned for ultrasound localization.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class UltrasoundLocalizer(nn.Module):
    """
    EfficientNet-B0 backbone with a regression head predicting (x, y).

    Architecture:
      - EfficientNet-B0 feature extractor (pretrained ImageNet)
      - Dropout for regularization (critical with <500 images)
      - Fully connected regression head → 2 outputs (normalized x, y)
    """

    def __init__(self, dropout: float = 0.4, freeze_backbone: bool = False):
        super().__init__()
        # Load pretrained backbone
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features  # 1280

        # Remove the classification head
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),          # outputs: (x_norm, y_norm)
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    def unfreeze_backbone(self):
        """Call after initial warm-up training to fine-tune the full network."""
        for p in self.features.parameters():
            p.requires_grad = True
        print("Backbone unfrozen — full fine-tuning enabled.")
