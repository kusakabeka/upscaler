"""
Training script for the ESPCN super-resolution model.

Usage
-----
    python train.py --data_dir data/train --val_dir data/val \\
                    --scale 2 --epochs 100 --batch_size 16 --lr 1e-3

The script saves checkpoints to ``checkpoints/`` and prints training /
validation loss + PSNR every epoch.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.srcnn import get_model
from utils.dataset import SRDataset, SRValidationDataset
from utils.image_utils import tensor_to_numpy
from utils.metrics import psnr


def parse_args():
    p = argparse.ArgumentParser(description="Train ESPCN super-resolution model")
    p.add_argument("--data_dir", type=str, default="data/train",
                   help="Path to training images")
    p.add_argument("--val_dir", type=str, default="data/val",
                   help="Path to validation images")
    p.add_argument("--scale", type=int, default=2, choices=[2, 4],
                   help="Upscaling factor")
    p.add_argument("--patch_size", type=int, default=96,
                   help="HR patch size for training crops")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=10,
                   help="Save checkpoint every N epochs")
    return p.parse_args()


def validate(model, val_loader, criterion, device):
    """Run one validation pass; return average loss and PSNR."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for lr_batch, hr_batch in val_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)
            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)
            total_loss += loss.item() * lr_batch.size(0)

            # compute PSNR per image
            for i in range(lr_batch.size(0)):
                sr_np = tensor_to_numpy(sr_batch[i])
                hr_np = tensor_to_numpy(hr_batch[i])
                total_psnr += psnr(sr_np, hr_np)
                count += 1

    avg_loss = total_loss / max(count, 1)
    avg_psnr = total_psnr / max(count, 1)
    return avg_loss, avg_psnr


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Scale factor: x{args.scale}")

    # ---- Datasets ----
    train_ds = SRDataset(
        root_dir=args.data_dir,
        scale_factor=args.scale,
        patch_size=args.patch_size,
        augment=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )

    val_loader = None
    if os.path.isdir(args.val_dir) and len(os.listdir(args.val_dir)) > 0:
        val_ds = SRValidationDataset(
            root_dir=args.val_dir,
            scale_factor=args.scale,
            patch_size=args.patch_size,
        )
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"[INFO] Training samples: {len(train_ds)}")

    # ---- Model, loss, optimizer ----
    model = get_model(scale_factor=args.scale, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {total_params:,}")

    # ---- Training loop ----
    best_val_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for lr_batch, hr_batch in train_loader:
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            sr_batch = model(lr_batch)
            loss = criterion(sr_batch, hr_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * lr_batch.size(0)

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_ds)
        elapsed = time.time() - t0

        msg = (
            f"Epoch [{epoch}/{args.epochs}]  "
            f"Train Loss: {avg_train_loss:.6f}  "
            f"Time: {elapsed:.1f}s"
        )

        # ---- Validation ----
        if val_loader is not None:
            val_loss, val_psnr = validate(model, val_loader, criterion, device)
            msg += f"  |  Val Loss: {val_loss:.6f}  Val PSNR: {val_psnr:.2f} dB"
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                path = os.path.join(args.checkpoint_dir, f"best_x{args.scale}.pth")
                torch.save(model.state_dict(), path)
                msg += "  *best*"

        print(msg)

        # ---- Periodic checkpoint ----
        if epoch % args.save_every == 0:
            path = os.path.join(
                args.checkpoint_dir, f"espcn_x{args.scale}_epoch{epoch}.pth"
            )
            torch.save(model.state_dict(), path)

    # ---- Final save ----
    final_path = os.path.join(args.checkpoint_dir, f"espcn_x{args.scale}_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[DONE] Final model saved to {final_path}")
    if best_val_psnr > 0:
        print(f"[DONE] Best validation PSNR: {best_val_psnr:.2f} dB")


if __name__ == "__main__":
    main()
