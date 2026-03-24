"""
Image loading, preprocessing and saving helpers.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array (H, W, 3), uint8."""
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_image(img: np.ndarray, path: str) -> str:
    """Save RGB numpy array to file. Returns the absolute path."""
    path = str(Path(path).resolve())
    Image.fromarray(img).save(path)
    return path


def numpy_to_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert HWC uint8 numpy image to CHW float32 tensor in [0, 1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)  # add batch dim


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor (or BCHW) to HWC uint8 numpy."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def resize_bicubic(img: np.ndarray, scale: int) -> np.ndarray:
    """Upscale image using bicubic interpolation (baseline comparison)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
