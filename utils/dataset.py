"""
Dataset utilities for super-resolution training.

The dataset takes high-resolution images, extracts random crops, and creates
paired (LR, HR) samples by down-scaling the crop with bicubic interpolation.
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Supported image extensions
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMG_EXTENSIONS


class SRDataset(Dataset):
    """Super-Resolution dataset that generates (LR, HR) pairs on the fly.

    Parameters
    ----------
    root_dir : str
        Directory with high-resolution images.
    scale_factor : int
        Down-sampling factor (2 or 4).
    patch_size : int
        Size of the HR crop (must be divisible by scale_factor).
    augment : bool
        Apply random horizontal/vertical flips.
    """

    def __init__(
        self,
        root_dir: str,
        scale_factor: int = 2,
        patch_size: int = 96,
        augment: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.augment = augment

        self.image_paths = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if is_image_file(f)
        )

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {root_dir}. "
                f"Supported formats: {IMG_EXTENSIONS}"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # ---- random crop of size patch_size x patch_size ----
        w, h = img.size
        ps = self.patch_size
        if w < ps or h < ps:
            img = img.resize(
                (max(w, ps), max(h, ps)), Image.BICUBIC
            )
            w, h = img.size

        left = random.randint(0, w - ps)
        top = random.randint(0, h - ps)
        hr_patch = img.crop((left, top, left + ps, top + ps))

        # ---- augmentation ----
        if self.augment:
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)

        # ---- create LR by down-sampling then up-sampling is NOT done ----
        # We keep LR at its native small size; the model up-samples.
        lr_size = ps // self.scale_factor
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)

        hr_tensor = self.to_tensor(hr_patch)   # (3, ps, ps)
        lr_tensor = self.to_tensor(lr_patch)   # (3, lr_size, lr_size)

        return lr_tensor, hr_tensor


class SRValidationDataset(Dataset):
    """Validation dataset — centre-crops full images, no augmentation."""

    def __init__(
        self,
        root_dir: str,
        scale_factor: int = 2,
        patch_size: int = 256,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size

        self.image_paths = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if is_image_file(f)
        )

        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # centre crop
        w, h = img.size
        ps = self.patch_size
        if w < ps or h < ps:
            img = img.resize((max(w, ps), max(h, ps)), Image.BICUBIC)
            w, h = img.size
        left = (w - ps) // 2
        top = (h - ps) // 2
        hr_patch = img.crop((left, top, left + ps, top + ps))

        lr_size = ps // self.scale_factor
        lr_patch = hr_patch.resize((lr_size, lr_size), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch)
