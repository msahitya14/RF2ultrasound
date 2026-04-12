import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np


def load_echocare_backbone(checkpoint_path, in_channels=3, feature_size=128):
    encoder = SwinTransformer(
        in_chans=in_channels,
        embed_dim=feature_size,
        window_size=[8] * 2,
        patch_size=[2] * 2,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        use_checkpoint=True,
        spatial_dims=2,
        use_v2=True
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        state_dict.pop("mask_token", None)  # remove if present
        encoder.load_state_dict(state_dict, strict=True)
        print("Loaded EchoCare pretrained weights")
    else:
        print("No checkpoint — using random init (not recommended)")

    return encoder

def main():
    x = torch.rand(1, 3, 256, 256)
    backbone = load_echocare_backbone(checkpoint_path="./echocare_encoder.pth")
    out = backbone(x)
    print(type(out))
    print([x_out.shape for x_out in out])


if __name__ == "__main__":
    main()

