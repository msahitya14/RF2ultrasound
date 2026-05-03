import torch
from monai.networks.nets.swin_unetr import SwinTransformer
import re
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms as T
import torch.nn as nn
checkpoint_path = "./echocare_encoder.pth"

def parse_filename(filename):
    # Match x and y values — m prefix means negative
    pattern = r'_x(m?)(\d+)_(\d+)_y(m?)(\d+)_(\d+)'
    match = re.search(pattern, filename)

    if not match:
        print(f"WARNING: Could not parse {filename}, skipping")
        return None

    x_neg, x_int, x_dec, y_neg, y_int, y_dec = match.groups()

    delta_x = float(f"{x_int}.{x_dec}")

    if x_neg == "m":
        delta_x = -delta_x

    return delta_x

def set_backbone_frozen(model, frozen: bool):
    for param in model.backbone.parameters():
        param.requires_grad = not frozen
    print(f"Backbone {'frozen' if frozen else 'unfrozen'}")

class ImageDataset(Dataset):
    def __init__(self, csv_file):
        # Load the CSV file once during initialization
        self.data = pd.read_csv(csv_file)
        self.transform = T.Compose([
                T.Resize((224, 224)),
                T.Grayscale(num_output_channels=3),
                T.ToTensor()
            ])

        # now compute stats on cleaned data
        self.min = self.data["delta_x"].quantile(0.2)
        self.max = self.data["delta_x"].quantile(0.8)

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a single row at the given index
        # Use .iloc to access the specific row
        sample = self.data.iloc[idx]

        # Separate features (X) and label (y)
        # Convert them to tensors (ensure they are numeric types)
        image = Image.open(sample["image_path"])
        image = self.transform(image)
        delta_x = sample["delta_x"]
        label = torch.tensor(delta_x, dtype = torch.float32)

        label = (label - self.min) / (self.max - self.min)
        label = label * 2 - 1
        label = torch.clamp(label, -1, 1)
        return image, label

class Regression(nn.Module):
    def __init__(self, backbone):
      super().__init__()
      self.backbone = backbone

      self.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # [1, 2048, 8, 8] → [1, 2048, 1, 1]
        nn.Flatten(),             # → [1, 2048]
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1)         # → [1, 1]  (single tilt value)
      )

    def forward(self, x):
      features = self.backbone(x)
      x = features[-1] # use more features if needed
      return self.head(x)
    

def train(model, train_loader, val_loader, num_epochs_frozen=5, num_epochs_finetune=10, device='cuda'):
    model = model.to(device)
    criterion = nn.SmoothL1Loss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss = 0

        with torch.set_grad_enabled(train):
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                if train:
                    optimizer.zero_grad()

                preds = model(images)
                loss = criterion(preds, labels)

                if train:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    # ── Phase 1: head only ──────────────────────────────────────────
    set_backbone_frozen(model, frozen=True)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

    for epoch in range(num_epochs_frozen):
        train_loss = run_epoch(train_loader, train=True)
        val_loss   = run_epoch(val_loader,   train=False)
        print(f"[Frozen]   Epoch {epoch+1}/{num_epochs_frozen} — Train: {train_loss:.4f}  Val: {val_loss:.4f}")

    # ── Phase 2: full fine-tune ─────────────────────────────────────
    set_backbone_frozen(model, frozen=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs_finetune)

    best_val_loss = float('inf')

    for epoch in range(num_epochs_finetune):
        train_loss = run_epoch(train_loader, train=True)
        val_loss   = run_epoch(val_loader,   train=False)
        scheduler.step()

        print(f"[Finetune] Epoch {epoch+1}/{num_epochs_finetune} — Train: {train_loss:.4f}  Val: {val_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        # save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ↳ saved checkpoint (val loss: {val_loss:.4f})")

    return model

def evaluate(model, test_loader, device='cuda', label_scale=1.0):
    model = model.to(device)
    model.eval()

    criterion = nn.SmoothL1Loss()

    all_preds  = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            preds = model(images)
            loss  = criterion(preds, labels)
            total_loss += loss.item()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds  = torch.cat(all_preds)  * label_scale  # scale back to degrees if normalized
    all_labels = torch.cat(all_labels) * label_scale

    avg_loss = total_loss / len(test_loader)
    mae      = (all_preds - all_labels).abs().mean().item()
    rmse     = ((all_preds - all_labels) ** 2).mean().sqrt().item()

    print(f"Test Loss (SmoothL1): {avg_loss:.4f}")
    print(f"MAE:                  {mae:.2f}°")
    print(f"RMSE:                 {rmse:.2f}°")

    return {
        'loss':      avg_loss,
        'mae':       mae,
        'rmse':      rmse,
        'preds':     all_preds,
        'labels':    all_labels,
    }

def main():
    encoder = SwinTransformer(
        in_chans=3,
        embed_dim=128,
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

    df = pd.DataFrame(columns=["image_path", "delta_x"])
    with os.scandir("./Images") as entries:
        for entry in entries:
            delta_x = parse_filename(entry.name)
            df.loc[len(df)] = [f"./Images/{entry.name}", delta_x]
    df.to_csv("dataset.csv", index=False)

    dataset = ImageDataset(csv_file="dataset.csv")
    max = dataset.max
    min = dataset.min
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_data, batch_size = 16, shuffle = True) # Reduced batch size from 32 to 8
    val_loader = DataLoader(val_data, batch_size = 4, shuffle = False) # Reduced batch size from 8 to 4
    test_loader = DataLoader(test_data, batch_size = 4, shuffle = False) # Reduced batch size from 8 to 4

    print("Train data length:", len(train_data))
    print("Test data length:", len(test_data))
    print("Val data length:", len(val_data))

    model = Regression(encoder)
    train(model, train_loader, val_loader)

    model.load_state_dict(torch.load('best_model.pth'))
    results = evaluate(model, test_loader, device='cuda', label_scale=1.0)
    print(f"\n{'Label':>10}  {'Prediction':>10}  {'Error':>10}")
    print("-" * 36)
    wrong_direction = 0
    for label, pred in zip(results['labels'], results['preds']):
        label_val = label.item()
        pred_val  = (pred.item() + 1) / 2 * (max - min) + min
        error_val = pred_val - label_val
        if (pred_val > 0 and label_val < 0) or (pred_val < 0 and label_val > 0):
            wrong_direction += 1
        print(f"{label_val:>10.2f}°  {pred_val:>10.2f}°  {error_val:>+10.2f}°")

    print()
    print("Wrong direction:", wrong_direction)
    
if __name__ == "__main__":
    main()



