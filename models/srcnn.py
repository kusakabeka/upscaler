"""
SRCNN / ESPCN-inspired Super-Resolution CNN model.

Architecture overview
---------------------
The model follows a three-stage pipeline inspired by the classic SRCNN paper
(Dong et al., 2014) combined with the sub-pixel convolution idea from ESPCN
(Shi et al., 2016):

1. Feature extraction – a convolutional layer that maps the input image into
   a high-dimensional feature space.
2. Non-linear mapping – one or more convolutional layers that learn the
   mapping between low-resolution and high-resolution feature representations.
3. Up-sampling / reconstruction – a PixelShuffle layer that rearranges the
   feature channels into spatial resolution, producing the final HR output.

Using PixelShuffle (sub-pixel convolution) instead of a deconvolution or
bicubic pre-upscaling keeps the network efficient: the model operates entirely
in the low-resolution space and only upscales at the very end.
"""

import torch
import torch.nn as nn


class ESPCN(nn.Module):
    """Efficient Sub-Pixel Convolutional Neural Network for super-resolution.

    Parameters
    ----------
    scale_factor : int
        Upscaling factor (2 or 4).
    num_channels : int
        Number of input / output image channels (default 3 for RGB).
    """

    def __init__(self, scale_factor: int = 2, num_channels: int = 3):
        super().__init__()
        self.scale_factor = scale_factor

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.non_linear_mapping = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Sub-pixel convolution: outputs (num_channels * scale^2) channels,
        # then PixelShuffle rearranges them into spatial dimensions.
        self.reconstruction = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Low-resolution image batch, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Super-resolved image batch, shape (B, C, H*scale, W*scale).
        """
        out = self.feature_extraction(x)
        out = self.non_linear_mapping(out)
        out = self.reconstruction(out)
        return out


def get_model(scale_factor: int = 2, device: str = "cpu") -> ESPCN:
    """Helper: create model and move to device."""
    model = ESPCN(scale_factor=scale_factor)
    return model.to(device)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for sf in (2, 4):
        net = get_model(scale_factor=sf)
        dummy = torch.randn(1, 3, 64, 64)
        out = net(dummy)
        print(f"scale x{sf}: input {dummy.shape} -> output {out.shape}")
        total_params = sum(p.numel() for p in net.parameters())
        print(f"  Total parameters: {total_params:,}")
