"""
predict.py — Run inference on one or more images to get (x, y) in degrees.

Usage:
    # Single image
    python predict.py --checkpoint checkpoints/best_model.pt \
                      --image frame_20260408_064836_9137_0015_x6_710_y6_463.png

    # Folder of images (also computes error if filenames contain ground truth)
    python predict.py --checkpoint checkpoints/best_model.pt \
                      --image_dir /path/to/test_images
"""

import os
import argparse
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from model import UltrasoundLocalizer
from dataset import parse_xy_from_filename, denormalize_x, denormalize_y


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = UltrasoundLocalizer(dropout=0.0)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  "
          f"val_err={ckpt['val_err_deg']:.3f}°")
    return model


def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])


def predict_single(model, transform, image_path: str, device: torch.device):
    img = Image.open(image_path).convert('RGB')
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(inp).squeeze(0).cpu()
    x_deg = denormalize_x(pred_norm[0:1]).item()
    y_deg = denormalize_y(pred_norm[1:2]).item()
    return x_deg, y_deg


def main(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else 'cpu'
    )
    model     = load_model(args.checkpoint, device)
    transform = get_transform(args.img_size)

    if args.image:
        x_pred, y_pred = predict_single(model, transform, args.image, device)
        print(f"\nPrediction for: {os.path.basename(args.image)}")
        print(f"  x = {x_pred:+.3f}°")
        print(f"  y = {y_pred:+.3f}°")
        try:
            x_gt, y_gt = parse_xy_from_filename(args.image)
            err = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
            print(f"  Ground truth: x={x_gt:+.3f}°  y={y_gt:+.3f}°")
            print(f"  Euclidean error: {err:.3f}°")
        except ValueError:
            pass

    elif args.image_dir:
        files = sorted([f for f in os.listdir(args.image_dir)
                        if f.lower().endswith('.png')])
        results, errors = [], []
        print(f"\n{'Filename':<60} {'x_pred':>8} {'y_pred':>8} {'err':>8}")
        print('-' * 90)
        for fname in files:
            fpath = os.path.join(args.image_dir, fname)
            x_pred, y_pred = predict_single(model, transform, fpath, device)
            row = dict(filename=fname,
                       x_pred=round(x_pred, 3), y_pred=round(y_pred, 3))
            try:
                x_gt, y_gt = parse_xy_from_filename(fname)
                err = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
                row.update(x_gt=x_gt, y_gt=y_gt, err_deg=round(err, 3))
                errors.append(err)
                print(f"{fname:<60} {x_pred:>+8.3f} {y_pred:>+8.3f} {err:>7.3f}°")
            except ValueError:
                print(f"{fname:<60} {x_pred:>+8.3f} {y_pred:>+8.3f} {'N/A':>8}")
            results.append(row)

        if errors:
            print(f"\nMean Euclidean error : {np.mean(errors):.3f}°")
            print(f"Median               : {np.median(errors):.3f}°")
            print(f"95th percentile      : {np.percentile(errors, 95):.3f}°")

        out = os.path.join(args.image_dir, 'predictions.json')
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_model.pt')
    parser.add_argument('--image',      default=None,
                        help='Path to a single image')
    parser.add_argument('--image_dir',  default=None,
                        help='Folder of images to evaluate')
    parser.add_argument('--img_size',   type=int, default=224)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        raise ValueError("Provide --image or --image_dir")
    main(args)