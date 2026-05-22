"""
train.py — Two-phase training for ultrasound (x, y) localization.

Phase 1 (warm-up):  Backbone frozen, only train the regression head.
Phase 2 (fine-tune): Unfreeze backbone, train end-to-end with lower LR.

Usage:
    python train.py --image_dir /path/to/images --output_dir ./checkpoints

    # Resume from a checkpoint, run both phases:
    python train.py --image_dir /path/to/images --resume ./checkpoints/best_model.pt

    # Load weights and skip straight to Phase 2:
    python train.py --image_dir /path/to/images --resume ./checkpoints/best_model.pt --resume_phase finetune
"""

import os
import argparse
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UltrasoundDataset, TransformDataset, get_transforms, make_splits
from model import UltrasoundLocalizer


def euclidean_error_deg(preds_norm, targets_norm, dataset):
    """Mean Euclidean distance error in degrees (denormalized)."""
    preds_deg   = dataset.denormalize(preds_norm.detach().cpu())
    targets_deg = dataset.denormalize(targets_norm.detach().cpu())
    return torch.sqrt(((preds_deg - targets_deg) ** 2).sum(dim=1)).mean().item()


def load_checkpoint(path, model, device, optimizer=None):
    """Load weights (and optionally optimizer state) from a checkpoint.

    Args:
        path:      Path to the .pt checkpoint file.
        model:     The model to load weights into.
        device:    Device to map tensors onto.
        optimizer: If provided and the checkpoint contains optimizer_state,
                   that state is restored too (useful for resuming mid-phase).

    Returns:
        best_val_err (float) — the val_err_deg stored in the checkpoint,
        or inf if the key is absent (e.g. a checkpoint from an older run).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    best = ckpt.get('val_err_deg', float('inf'))
    print(f"  Loaded checkpoint : {path}")
    print(f"  └─ epoch={ckpt.get('epoch', '?')}  "
          f"phase={ckpt.get('phase', '?')}  "
          f"val_err={best:.4f}°")
    return best


def run_epoch(model, loader, optimizer, criterion, device, dataset,
              train=True):
    model.train() if train else model.eval()
    total_loss, total_err, n = 0.0, 0.0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)

            if train:
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                preds = model(imgs)
                loss = criterion(preds, targets)

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_err  += euclidean_error_deg(preds, targets, dataset) * bs
            n += bs

    return total_loss / n, total_err / n


def train_phase(model, train_loader, val_loader, optimizer, scheduler,
                criterion, device, dataset, epochs, phase_name,
                output_dir, best_val_err, history):
    print(f"\n{'='*52}")
    print(f"  {phase_name}  ({epochs} epochs)")
    print(f"{'='*52}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_err = run_epoch(model, train_loader, optimizer,
                                           criterion, device, dataset, train=True)
        val_loss, val_err     = run_epoch(model, val_loader,   None,
                                           criterion, device, dataset, train=False)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']

        row = dict(phase=phase_name, epoch=epoch,
                   train_loss=round(train_loss, 6),
                   val_loss=round(val_loss, 6),
                   train_err_deg=round(train_err, 4),
                   val_err_deg=round(val_err, 4),
                   lr=lr)
        history.append(row)

        flag = ''
        if val_err < best_val_err:
            best_val_err = val_err
            ckpt_path = os.path.join(output_dir, 'best_model.pt')
            torch.save({
                'epoch':           epoch,
                'phase':           phase_name,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),   # ← added
                'val_err_deg':     val_err,
            }, ckpt_path)
            flag = '  ← best'

        elapsed = time.time() - t0
        print(f"  [{epoch:3d}/{epochs}] "
              f"loss={train_loss:.4f}/{val_loss:.4f}  "
              f"err={train_err:.3f}/{val_err:.3f}°  "
              f"lr={lr:.2e}  {elapsed:.1f}s{flag}")

    return best_val_err


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    # ── Dataset & splits ────────────────────────────────────────────────────
    full_ds = UltrasoundDataset(args.image_dir)

    train_split, val_split, test_split = make_splits(
        full_ds, val_frac=0.15, test_frac=0.10)

    train_tf, val_tf = get_transforms(img_size=args.img_size, augment=True)

    train_ds = TransformDataset(train_split, train_tf)
    val_ds   = TransformDataset(val_split,   val_tf)
    test_ds  = TransformDataset(test_split,  val_tf)

    print(f"Splits — train: {len(train_ds)}, val: {len(val_ds)}, "
          f"test: {len(test_ds)}")

    loader_kwargs = dict(batch_size=args.batch_size,
                         num_workers=args.workers,
                         pin_memory=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    # ── Model ───────────────────────────────────────────────────────────────
    model     = UltrasoundLocalizer(dropout=args.dropout,
                                    freeze_backbone=True).to(device)
    criterion = nn.MSELoss()
    history   = []
    best_val_err = float('inf')

    # ── Resume / fine-tune from an existing checkpoint ──────────────────────
    # Weights are loaded here, before any optimizer is built, so both phases
    # start from the resumed weights regardless of --resume_phase.
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        best_val_err = load_checkpoint(args.resume, model, device)

    # ── Phase 1: warm-up (head only) ────────────────────────────────────────
    # Skipped when --resume_phase finetune is set (jump straight to Phase 2).
    skip_warmup = bool(args.resume and args.resume_phase == 'finetune')

    if not skip_warmup:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr_warmup, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=1e-6)

        best_val_err = train_phase(
            model, train_loader, val_loader, optimizer, scheduler,
            criterion, device, full_ds, args.warmup_epochs,
            'Phase1-WarmUp', args.output_dir, best_val_err, history)
    else:
        print("Skipping Phase 1 (--resume_phase finetune) — "
              "jumping straight to fine-tune.")

    # ── Phase 2: full fine-tune ──────────────────────────────────────────────
    model.unfreeze_backbone()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_finetune, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs, eta_min=1e-7)

    best_val_err = train_phase(
        model, train_loader, val_loader, optimizer, scheduler,
        criterion, device, full_ds, args.finetune_epochs,
        'Phase2-FineTune', args.output_dir, best_val_err, history)

    # ── Test evaluation ──────────────────────────────────────────────────────
    print("\n--- Test Set Evaluation (best checkpoint) ---")
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pt'),
                      map_location=device)
    model.load_state_dict(ckpt['model_state'])
    _, test_err = run_epoch(model, test_loader, None, criterion,
                             device, full_ds, train=False)
    print(f"  Test Euclidean error: {test_err:.3f} deg")

    # ── Save outputs ─────────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val error: {best_val_err:.3f}°")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ultrasound localizer')
    parser.add_argument('--image_dir',       required=True)
    parser.add_argument('--output_dir',      default='./checkpoints')
    parser.add_argument('--img_size',        type=int,   default=224)
    parser.add_argument('--batch_size',      type=int,   default=16)
    parser.add_argument('--warmup_epochs',   type=int,   default=20)
    parser.add_argument('--finetune_epochs', type=int,   default=60)
    parser.add_argument('--lr_warmup',       type=float, default=1e-3)
    parser.add_argument('--lr_finetune',     type=float, default=1e-4)
    parser.add_argument('--dropout',         type=float, default=0.4)
    parser.add_argument('--workers',         type=int,   default=0,
                        help='DataLoader workers. Keep 0 on macOS/Python 3.13.')
    # ── Checkpoint resume ────────────────────────────────────────────────────
    parser.add_argument('--resume',          default=None,
                        help='Path to a .pt checkpoint to load weights from '
                             'before training begins.')
    parser.add_argument('--resume_phase',    default='both',
                        choices=['both', 'finetune'],
                        help='"both" runs Phase 1 then Phase 2 starting from '
                             'the checkpoint weights. '
                             '"finetune" skips Phase 1 entirely.')
    args = parser.parse_args()
    main(args)