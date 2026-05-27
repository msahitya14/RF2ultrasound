"""
Andrew's Swin Transformer regression model.
Predicts a single normalized x-tilt value in [-1, 1].
"""

import torch.nn as nn
from monai.networks.nets.swin_unetr import SwinTransformer
from torchvision import transforms


class AndrewRegression(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features[-1])


def build_model() -> AndrewRegression:
    backbone = SwinTransformer(
        in_chans=3,
        embed_dim=128,
        window_size=[8, 8],
        patch_size=[2, 2],
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        use_checkpoint=True,
        spatial_dims=2,
        use_v2=True,
    )
    return AndrewRegression(backbone)


def get_transform(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
