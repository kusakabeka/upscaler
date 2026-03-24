"""
Quick test script — validates the full pipeline without needing real data.

Creates a synthetic test image, runs inference, computes metrics, and
optionally runs a short training loop on synthetic patches.

Usage
-----
    python test_pipeline.py              # full pipeline test
    python test_pipeline.py --no-train   # skip training test
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

# ---- helpers ----
def make_synthetic_image(width=128, height=128) -> np.ndarray:
    """Generate a colourful gradient + circle test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            img[y, x] = [
                int(255 * x / width),
                int(255 * y / height),
                128,
            ]
    # add a circle
    cy, cx, r = height // 2, width // 2, min(height, width) // 4
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    img[mask] = [255, 255, 0]
    return img


def test_model():
    """Test model creation and forward pass."""
    print("=" * 60)
    print("TEST 1: Model creation & forward pass")
    print("=" * 60)
    from models.srcnn import get_model

    for scale in (2, 4):
        model = get_model(scale_factor=scale)
        x = torch.randn(1, 3, 32, 32)
        y = model(x)
        expected_h = 32 * scale
        assert y.shape == (1, 3, expected_h, expected_h), (
            f"Wrong output shape: {y.shape}"
        )
        params = sum(p.numel() for p in model.parameters())
        print(f"  x{scale}: input {x.shape} -> output {y.shape}  "
              f"({params:,} params) ✓")
    print()


def test_image_utils():
    """Test image loading/saving helpers."""
    print("=" * 60)
    print("TEST 2: Image utilities")
    print("=" * 60)
    from utils.image_utils import (
        load_image, save_image, numpy_to_tensor,
        tensor_to_numpy, resize_bicubic,
    )

    img = make_synthetic_image(64, 64)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        Image.fromarray(img).save(f.name)
        loaded = load_image(f.name)
        assert loaded.shape == (64, 64, 3), f"Bad shape: {loaded.shape}"
        print(f"  load_image: {loaded.shape} ✓")

        t = numpy_to_tensor(loaded)
        assert t.shape == (1, 3, 64, 64), f"Bad tensor shape: {t.shape}"
        print(f"  numpy_to_tensor: {t.shape} ✓")

        back = tensor_to_numpy(t)
        assert back.shape == (64, 64, 3)
        print(f"  tensor_to_numpy: {back.shape} ✓")

        bic = resize_bicubic(loaded, 2)
        assert bic.shape == (128, 128, 3)
        print(f"  resize_bicubic x2: {bic.shape} ✓")

        out_path = save_image(bic, f.name.replace(".png", "_out.png"))
        assert os.path.isfile(out_path)
        print(f"  save_image: {out_path} ✓")

        os.unlink(f.name)
        os.unlink(out_path)
    print()


def test_metrics():
    """Test PSNR and SSIM computation."""
    print("=" * 60)
    print("TEST 3: Quality metrics")
    print("=" * 60)
    from utils.metrics import psnr, ssim, compute_metrics

    a = make_synthetic_image(64, 64)
    # identical images → PSNR = inf
    assert psnr(a, a) == float("inf"), "PSNR of identical images should be inf"
    print("  PSNR (identical) = inf ✓")

    b = a.copy()
    b = np.clip(b.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    p = psnr(a, b)
    assert 20 < p < 50, f"Unexpected PSNR: {p}"
    print(f"  PSNR (slightly noisy) = {p:.2f} dB ✓")

    s = ssim(a, a)
    assert s > 0.99, f"SSIM of identical should be ~1.0, got {s}"
    print(f"  SSIM (identical) = {s:.4f} ✓")

    m = compute_metrics(a, b)
    print(f"  compute_metrics: {m} ✓")
    print()


def test_inference():
    """Test full inference pipeline."""
    print("=" * 60)
    print("TEST 4: Inference pipeline")
    print("=" * 60)
    from models.srcnn import get_model
    from utils.image_utils import numpy_to_tensor, tensor_to_numpy

    img = make_synthetic_image(48, 48)
    model = get_model(scale_factor=2)
    model.eval()

    lr = numpy_to_tensor(img)
    with torch.no_grad():
        sr = model(lr)
    result = tensor_to_numpy(sr)
    assert result.shape == (96, 96, 3), f"Bad SR shape: {result.shape}"
    print(f"  48×48 → {result.shape[1]}×{result.shape[0]} ✓")
    print()


def test_training(num_epochs=3):
    """Run a tiny training loop to verify the training pipeline."""
    print("=" * 60)
    print("TEST 5: Training loop (mini)")
    print("=" * 60)

    # create temp directory with a few synthetic images
    tmpdir = tempfile.mkdtemp()
    for i in range(8):
        img = make_synthetic_image(128, 128)
        Image.fromarray(img).save(os.path.join(tmpdir, f"img_{i:03d}.png"))

    from utils.dataset import SRDataset
    from models.srcnn import get_model

    ds = SRDataset(root_dir=tmpdir, scale_factor=2, patch_size=64, augment=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

    model = get_model(scale_factor=2)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for lr_batch, hr_batch in loader:
            sr = model(lr_batch)
            loss = criterion(sr, hr_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"  Epoch {epoch}/{num_epochs}  loss={avg:.6f}")

    # verify checkpoint save/load
    ckpt_path = os.path.join(tmpdir, "test_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    model2 = get_model(scale_factor=2)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"  Checkpoint save/load ✓")

    # cleanup
    import shutil
    shutil.rmtree(tmpdir)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-train", action="store_true")
    args = parser.parse_args()

    test_model()
    test_image_utils()
    test_metrics()
    test_inference()

    if not args.no_train:
        test_training()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
