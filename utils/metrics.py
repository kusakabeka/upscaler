"""
Image quality metrics for super-resolution evaluation.

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
"""

import numpy as np
import torch


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 255.0) -> float:
    """Compute PSNR between two uint8 images of the same shape."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)


def ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    max_val: float = 255.0,
    window_size: int = 11,
) -> float:
    """Simplified SSIM computed on grayscale conversion.

    For a rigorous implementation use skimage.metrics.structural_similarity;
    this version avoids the extra dependency while giving reasonable results.
    """
    try:
        from skimage.metrics import structural_similarity
        # Convert to grayscale if RGB
        if img1.ndim == 3:
            import cv2
            g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            g1, g2 = img1, img2
        return structural_similarity(g1, g2, data_range=max_val)
    except ImportError:
        # Fallback: simplified SSIM
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = img1.var()
        sigma2_sq = img2.var()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
        return float(numerator / denominator)


def compute_metrics(sr: np.ndarray, hr: np.ndarray) -> dict:
    """Return dict with PSNR and SSIM between super-resolved and ground-truth."""
    return {
        "psnr": round(psnr(sr, hr), 2),
        "ssim": round(ssim(sr, hr), 4),
    }
