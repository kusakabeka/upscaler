"""
Inference script — apply super-resolution to a single image.

Usage
-----
    python infer.py --input photo.jpg --scale 2 \\
                    --checkpoint checkpoints/best_x2.pth \\
                    --output outputs/photo_sr.png

If ``--checkpoint`` is omitted the model runs with random (untrained) weights
— useful for verifying the pipeline before training.
"""

import argparse
import os
import time

import torch
from PIL import Image

from models.srcnn import get_model
from utils.image_utils import (
    load_image,
    save_image,
    numpy_to_tensor,
    tensor_to_numpy,
    resize_bicubic,
)
from utils.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Super-resolution inference")
    p.add_argument("--input", type=str, required=True, help="Input image path")
    p.add_argument("--output", type=str, default=None, help="Output path")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--compare", action="store_true",
                   help="Also save bicubic upscaled version for comparison")
    return p.parse_args()


def super_resolve(
    image_path: str,
    scale: int = 2,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> tuple:
    """Run SR on a single image. Returns (sr_numpy, elapsed_seconds)."""
    img = load_image(image_path)
    lr_tensor = numpy_to_tensor(img).to(device)

    model = get_model(scale_factor=scale, device=device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[WARN] No checkpoint loaded — using random weights (demo mode)")

    model.eval()
    with torch.no_grad():
        t0 = time.time()
        sr_tensor = model(lr_tensor)
        elapsed = time.time() - t0

    sr_np = tensor_to_numpy(sr_tensor)
    return sr_np, elapsed


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    sr_img, elapsed = super_resolve(
        args.input, args.scale, args.checkpoint, device
    )
    print(f"[INFO] Inference time: {elapsed:.3f}s")
    print(f"[INFO] Output size: {sr_img.shape[1]}x{sr_img.shape[0]}")

    # Save output
    os.makedirs("outputs", exist_ok=True)
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"outputs/{stem}_x{args.scale}_sr.png"

    save_image(sr_img, args.output)
    print(f"[INFO] Saved to {args.output}")

    # Optional bicubic comparison
    if args.compare:
        lr_img = load_image(args.input)
        bicubic = resize_bicubic(lr_img, args.scale)
        bic_path = args.output.replace("_sr.", "_bicubic.")
        save_image(bicubic, bic_path)
        print(f"[INFO] Bicubic comparison saved to {bic_path}")

        metrics = compute_metrics(sr_img, bicubic)
        print(f"[INFO] SR vs Bicubic — PSNR: {metrics['psnr']} dB, SSIM: {metrics['ssim']}")


if __name__ == "__main__":
    main()
