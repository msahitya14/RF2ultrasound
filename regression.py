import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinTransformer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
import os
from PIL import Image
import re
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def parse_filename(filename):
    """
    Parses delta_x and delta_y from filenames like:
    frame_20260407_004352_2723_0006_xm3_629_y3_376
    
    xm3_629  → -3.629
    y3_376   → +3.376
    ym0_860  → -0.860
    """
    # Match x and y values — m prefix means negative
    pattern = r'_x(m?)(\d+)_(\d+)_y(m?)(\d+)_(\d+)'
    match = re.search(pattern, filename)

    if not match:
        print(f"WARNING: Could not parse {filename}, skipping")
        return None

    x_neg, x_int, x_dec, y_neg, y_int, y_dec = match.groups()

    delta_x = float(f"{x_int}.{x_dec}")
    delta_y = float(f"{y_int}.{y_dec}")

    if x_neg == "m":
        delta_x = -delta_x
    if y_neg == "m":
        delta_y = -delta_y

    return delta_x, delta_y


def build_csv_from_folder(folder_path, output_csv="data/labels.csv"):
    """
    Walks through a folder of images, parses (delta_x, delta_y) from
    each filename, and saves a CSV ready for NeckDataset.
    """
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    rows = []
    skipped = 0

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(valid_extensions):
            continue

        result = parse_filename(filename)
        if result is None:
            skipped += 1
            continue

        delta_x, delta_y = result
        image_path = os.path.join(folder_path, filename)
        rows.append({
            "image_path": image_path,
            "delta_x":    delta_x,
            "delta_y":    delta_y,
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Parsed {len(rows)} images, skipped {skipped}")
    print(f"Saved to {output_csv}")
    print(df.head())
    return df

# Adjust batch size and split accordingly to dataset size
def get_loaders():
    transform = T.Compose([
        T.Resize((640, 480)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor()
    ])
    dataset = datasets.ImageFolder(root = "Data", transform= transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size = int(train_size/5), shuffle = True)
    test_loader = DataLoader(test_data, batch_size = len(dataset) - int(train_size/5), shuffle = False)
    return (train_loader, test_loader)

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

class Regression(nn.Module):
    def __init__(self, backbone, feature_size=128):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

        fused_dim = 1024 + 2048  # = 3072

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)        # (Δx, Δy)
        )

    def forward(self, x):
        x_outs = self.backbone(x)   # list of 5 feature maps

        # Take last two: [1, 1024, 16, 16] and [1, 2048, 8, 8]
        f4 = self.pool(x_outs[-2]).flatten(1)   # (B, 1024)
        f5 = self.pool(x_outs[-1]).flatten(1)   # (B, 2048)

        fused = torch.cat([f4, f5], dim=1)      # (B, 3072)
        return self.head(fused)
    
class NeckDataset(Dataset):
    """
    CSV columns: image_path, delta_x, delta_y
    """
    def __init__(self, csv_path, augment=False, mean=None, std=None):
        self.df      = pd.read_csv(csv_path)
        self.augment = augment

        # Use provided mean/std, else safe default for grayscale ultrasound
        mean = mean or [0.5, 0.5, 0.5]
        std  = std  or [0.5, 0.5, 0.5]

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),              # [0,255] → [0,1]
            transforms.Normalize(mean, std),    # [0,1]   → ~[-1,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"])
        dx, dy = float(row["delta_x"]), float(row["delta_y"])

        if self.augment:
            image, dx, dy = self._augment(image, dx, dy)

        image = self.transform(image)
        label = torch.tensor([dx, dy], dtype=torch.float32)
        return image, label

    def _augment(self, image, dx, dy):
        # Spatial — label must mirror image
        if random.random() > 0.5:
            image = TF.hflip(image)
            dx = -dx

        if random.random() > 0.5:
            image = TF.vflip(image)
            dy = -dy

        # Intensity — no label change needed
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))

        return image, dx, dy
    
# ─────────────────────────────────────────────────────────
# 4. LOSS
# ─────────────────────────────────────────────────────────
def offset_loss(preds, targets):
    """
    Combined MSE + Euclidean distance loss.
    MSE: stable gradients early in training
    Euclidean: directly meaningful as spatial error
    """
    mse  = F.mse_loss(preds, targets)
    dist = torch.sqrt(((preds - targets) ** 2).sum(dim=1) + 1e-6).mean()
    return 0.5 * mse + 0.5 * dist


# ─────────────────────────────────────────────────────────
# 5. FREEZE CONTROL
# ─────────────────────────────────────────────────────────
def set_backbone_frozen(model, frozen: bool):
    for param in model.backbone.parameters():
        param.requires_grad = not frozen
    print(f"Backbone {'frozen' if frozen else 'unfrozen'}")


# ─────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────
def run_phase(model, train_loader, val_loader, epochs, optimizer, device, label):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val  = float("inf")
    print(f"\n── {label} ──")

    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        # Train
        model.train()
        train_losses = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss  = offset_loss(preds, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        dist_errors = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
                dist  = torch.sqrt(((preds - labels) ** 2).sum(dim=1))
                dist_errors.extend(dist.cpu().numpy())

        avg_dist = np.mean(dist_errors)

        # Save best model
        if avg_dist < best_val:
            best_val = avg_dist
            torch.save(model.state_dict(), "best_model.pth")

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train loss: {np.mean(train_losses):.4f} | "
                  f"Mean dist: {avg_dist:.4f} | "
                  f"Median dist: {np.median(dist_errors):.4f} | "
                  f"Best: {best_val:.4f}")

    return best_val


def train(model, train_loader, val_loader, device="cuda"):
    model = model.to(device)

    # Phase 1: frozen backbone — validate EchoCare features are useful
    set_backbone_frozen(model, frozen=True)
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=1e-3, weight_decay=1e-4
    )
    run_phase(model, train_loader, val_loader,
              epochs=5, optimizer=optimizer,
              device=device, label="Phase 1: head only (backbone frozen)")

    # Phase 2: unfreeze backbone — adapt to your neck feature
    set_backbone_frozen(model, frozen=False)
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(),     "lr": 5e-4},
        {"params": model.backbone.parameters(), "lr": 1e-4},  # very slow
    ], weight_decay=1e-4)
    run_phase(model, train_loader, val_loader,
              epochs=5, optimizer=optimizer,
              device=device, label="Phase 2: fine-tune backbone")

    # Load best weights from either phase
    model.load_state_dict(torch.load("best_model.pth"))
    print("\nLoaded best model weights")
    return model


def predict_folder(model, folder_path, device="cuda", mean=None, std=None):
    mean = mean or [0.5, 0.5, 0.5]
    std  = std  or [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Get all images in the folder
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [
        f for f in sorted(os.listdir(folder_path))
        if f.lower().endswith(valid_extensions)
    ]

    model.eval()
    with torch.no_grad():
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0).to(device)

            pred = model(image).squeeze(0).cpu().numpy()
            print(f"{filename} | delta_x: {pred[0]:+.3f}, delta_y: {pred[1]:+.3f}")


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ── Parse labels from image filenames → build CSVs ──
    df = build_csv_from_folder(
        folder_path="./Images",
        output_csv="./labels.csv"
    )
    print(df[["delta_x", "delta_y"]].describe())


    # Split into train/val
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv",     index=False)
    print(f"Train: {len(train_df)} images, Val: {len(val_df)} images")
    

    # ── Load backbone ──
    backbone = load_echocare_backbone(
        checkpoint_path="./echocare_encoder.pth"
    )
    model = Regression(backbone)

    # ── Datasets ──
    train_dataset = NeckDataset("data/train.csv", augment=True)
    val_dataset   = NeckDataset("data/val.csv",   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4)

    # ── Train ──
    print("Training:")
    model = train(model, train_loader, val_loader, device=DEVICE)

    # ── Load best weights ──
    model.load_state_dict(torch.load("best_model.pth"))
    print("Loaded best model for inference")

    # ── Predict on same image folder ──
    predict_folder(model, "./TestImages", device=DEVICE)


if __name__ == "__main__":
    main()

