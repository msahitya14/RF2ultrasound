"""
dataset.py — Parses filenames and builds a PyTorch Dataset for ultrasound localization.

Filename format examples:
  frame_..._x6_710_y6_463.png        →  x=+6.710°, y=+6.463°
  frame_..._xm0_644_ym0_457.png      →  x=-0.644°, y=-0.457°
  frame_..._xm21_457_y0_263.png      →  x=-21.457°, y=+0.263°

Normalization:
  Uses fixed physical bounds from the IMU sensor (DeviceOrientation API):
    x = e.beta  - baseBeta   →  range ±180°  → normalized to [-1, 1]
    y = e.gamma - baseGamma  →  range  ±90°  → normalized to [-1, 1]

  This is better than data-driven z-score for a small dataset because:
    - Stats don't shift between runs or if you collect more data
    - The model sees a consistent, symmetric input space
    - Denormalization is exact and physically meaningful
"""

import os
import re
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image

# Physical bounds (degrees) — derived from DeviceOrientation API spec:
#   beta  (x): device tilt front/back  → ±180°
#   gamma (y): device rotation l/r     →  ±90°
X_BOUND = 180.0
Y_BOUND = 90.0


def parse_xy_from_filename(fname: str):
    """
    Extracts x and y (in degrees) from a filename.
    'xm' prefix means negative:  xm21_457 → -21.457°
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r'_x(m?)(\d+)_(\d+)_y(m?)(\d+)_(\d+)', base)
    if not m:
        raise ValueError(f"Cannot parse x/y from filename: {fname}")
    x_sign = -1.0 if m.group(1) == 'm' else 1.0
    y_sign = -1.0 if m.group(4) == 'm' else 1.0
    x = x_sign * float(f"{m.group(2)}.{m.group(3)}")
    y = y_sign * float(f"{m.group(5)}.{m.group(6)}")
    return x, y


def normalize_x(x: float) -> float:
    """Map x angle from [-180, +180] → [-1, +1]"""
    return x / X_BOUND


def normalize_y(y: float) -> float:
    """Map y angle from [-90, +90] → [-1, +1]"""
    return y / Y_BOUND


def denormalize_x(x_norm: torch.Tensor) -> torch.Tensor:
    """Map x from [-1, +1] → [-180, +180]"""
    return x_norm * X_BOUND


def denormalize_y(y_norm: torch.Tensor) -> torch.Tensor:
    """Map y from [-1, +1] → [-90, +90]"""
    return y_norm * Y_BOUND


class UltrasoundDataset(Dataset):
    """
    Loads all PNG images from `image_dir`, parses (x, y) angles from filenames,
    and normalizes targets to [-1, 1] using fixed physical bounds.
    """

    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform  # usually None; applied via TransformDataset

        self.samples = []
        skipped = 0
        for fname in sorted(os.listdir(image_dir)):
            if not fname.lower().endswith('.png'):
                continue
            try:
                x, y = parse_xy_from_filename(fname)
                self.samples.append((os.path.join(image_dir, fname), x, y))
            except ValueError:
                print(f"  [skip] {fname}")
                skipped += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid images found in {image_dir}")

        xs = torch.tensor([s[1] for s in self.samples], dtype=torch.float32)
        ys = torch.tensor([s[2] for s in self.samples], dtype=torch.float32)

        print(f"Dataset: {len(self.samples)} images  ({skipped} skipped)")
        print(f"  x: range=[{xs.min():.2f}°, {xs.max():.2f}°]  "
              f"(normalized by ±{X_BOUND}°)")
        print(f"  y: range=[{ys.min():.2f}°, {ys.max():.2f}°]  "
              f"(normalized by ±{Y_BOUND}°)")

        # Warn if any values exceed the expected physical bounds
        if xs.abs().max() > X_BOUND:
            print(f"  ⚠ WARNING: x exceeds ±{X_BOUND}°: {xs.abs().max():.2f}°")
        if ys.abs().max() > Y_BOUND:
            print(f"  ⚠ WARNING: y exceeds ±{Y_BOUND}°: {ys.abs().max():.2f}°")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, x_raw, y_raw = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        x_norm = normalize_x(x_raw)
        y_norm = normalize_y(y_raw)
        return img, torch.tensor([x_norm, y_norm], dtype=torch.float32)

    def denormalize(self, preds: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions [-1,1] back to degrees."""
        x_deg = denormalize_x(preds[:, 0])
        y_deg = denormalize_y(preds[:, 1])
        return torch.stack([x_deg, y_deg], dim=1)


# ── Module-level class — required for DataLoader multiprocessing pickle ───────

class TransformDataset(Dataset):
    """Wraps a Subset and applies a transform. Defined at module level so
    DataLoader worker processes can pickle it (local classes cannot be pickled)."""

    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, target = self.subset[idx]   # img is PIL Image (no transform on base ds)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

# ─────────────────────────────────────────────────────────────────────────────


def get_transforms(img_size: int = 224, augment: bool = True):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def make_splits(dataset: UltrasoundDataset, val_frac=0.15, test_frac=0.10,
                seed=42):
    n = len(dataset)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    return random_split(dataset, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(seed))