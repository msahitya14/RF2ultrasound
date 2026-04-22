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
import math
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Project each scale to same dim before fusion
        self.proj2 = nn.Conv2d(512,  128, 1)   # f2
        self.proj3 = nn.Conv2d(1024, 128, 1)   # f3
        self.proj4 = nn.Conv2d(2048, 128, 1)   # f4

        fused_dim = 128 * 3  # 384

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        outs = self.backbone(x)

        # Multi-scale: use f2, f3, f4 — f2 retains more spatial info
        f2 = self.pool(self.proj2(outs[2])).flatten(1)  # (B, 128)
        f3 = self.pool(self.proj3(outs[3])).flatten(1)  # (B, 128)
        f4 = self.pool(self.proj4(outs[4])).flatten(1)  # (B, 128)

        fused = torch.cat([f2, f3, f4], dim=1)          # (B, 384)
        return self.head(fused)
    
class NeckDataset(Dataset):
    """
    CSV columns: image_path, delta_x, delta_y
 
    Label normalisation
    -------------------
    Pass label_mean / label_std (computed from the TRAIN split only) to both
    train and val datasets so targets live in ~[-2, 2] during training.
 
    Usage
    -----
        train_ds = NeckDataset("train.csv", augment=True)
        val_ds   = NeckDataset("val.csv",   augment=False,
                               label_mean=train_ds.label_mean,
                               label_std=train_ds.label_std)
    """
 
    def __init__(self, csv_path, augment=False,
                 image_mean=None, image_std=None,
                 label_mean=None, label_std=None):
 
        self.df      = pd.read_csv(csv_path)
        self.augment = augment
 
        # ── image normalisation ───────────────────────────────────────────────
        image_mean = image_mean or [0.5, 0.5, 0.5]
        image_std  = image_std  or [0.5, 0.5, 0.5]
 
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std),
        ])
 
        # ── label normalisation ───────────────────────────────────────────────
        # If not provided, compute from this CSV (should only be the train split)
        if label_mean is not None and label_std is not None:
            self.label_mean = torch.tensor(label_mean, dtype=torch.float32)
            self.label_std  = torch.tensor(label_std,  dtype=torch.float32)
        else:
            self.label_mean = torch.tensor(
                [self.df["delta_x"].mean(), self.df["delta_y"].mean()],
                dtype=torch.float32)
            self.label_std  = torch.tensor(
                [self.df["delta_x"].std(),  self.df["delta_y"].std()],
                dtype=torch.float32)
 
    # ── helpers ───────────────────────────────────────────────────────────────
 
    def normalize_label(self, label: torch.Tensor) -> torch.Tensor:
        return (label - self.label_mean) / self.label_std
 
    def denormalize_label(self, label: torch.Tensor) -> torch.Tensor:
        """Call this on model output to get back pixel units."""
        return label * self.label_std + self.label_mean
 
    # ── dataset protocol ─────────────────────────────────────────────────────
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        image  = Image.open(row["image_path"])
        dx, dy = float(row["delta_x"]), float(row["delta_y"])
 
        if self.augment:
            image, dx, dy = self._augment(image, dx, dy)
 
        image = self.transform(image)
        label = self.normalize_label(
            torch.tensor([dx, dy], dtype=torch.float32)
        )
        return image, label
 
    # ── augmentation ─────────────────────────────────────────────────────────
 
    def _augment(self, image, dx, dy):
        # Horizontal flip — label x inverts
        # (safe for ultrasound; probe left/right is arbitrary)
        if random.random() > 0.5:
            image = TF.hflip(image)
            dx = -dx
 
        # Vertical flip removed — ultrasound depth axis (near→far) must not flip
 
        # Small rotation — rotate the (dx, dy) vector by the same angle
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            rad   = math.radians(angle)
            dx, dy = (dx * math.cos(rad) - dy * math.sin(rad),
                      dx * math.sin(rad) + dy * math.cos(rad))
 
        # Intensity — gain / TGC variation common in ultrasound
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image,   random.uniform(0.8, 1.2))
 
        # Speckle-like noise
        tensor = TF.to_tensor(image)
        tensor = torch.clamp(tensor + torch.randn_like(tensor) * 0.02, 0, 1)
        image  = TF.to_pil_image(tensor)
 
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
        {"params": model.head.parameters(),     "lr": 1e-4},
        {"params": model.backbone.parameters(), "lr": 1e-6},  # very slow
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

def sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = load_echocare_backbone(
        checkpoint_path="./echocare_encoder.pth"
    )
    model = Regression(backbone)

    # ── Datasets ──
    train_dataset = NeckDataset("data/train.csv", augment=True)
    val_dataset   = NeckDataset("data/val.csv",   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=4)
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images[:1].to(device), labels[:1].to(device)

    feats, targets = [], []
    for images, labels in train_loader:
        feats.append(images.mean(dim=[2,3]).numpy())  # (B, 3) — just mean pixel per channel
        targets.append(labels.numpy())

    feats   = np.vstack(feats)
    targets = np.vstack(targets)

    scaler = StandardScaler()
    feats  = scaler.fit_transform(feats)

    reg = Ridge().fit(feats, targets)
    print("R² on train:", reg.score(feats, targets))

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
    sanity_check()
    

