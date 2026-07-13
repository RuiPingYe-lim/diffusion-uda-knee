from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class StrictBridgeUNet(nn.Module):
    """
    Strict-style bridge model for pseudo-paired BBDM-style approximation.

    Input: bridge state x_t and timestep t.
    Output: prediction of BBDM bridge target bb_t.

    Bridge target definition:
      bb_t = m_t * (x_B - x_A) + sqrt(delta_t) * eps

    This keeps single-channel MRI input and pair-based bridge workflow.
    When pair_mode=label_random, data assumption remains pseudo-paired/
    class-consistent rather than official fully paired BBDM.
    """

    def __init__(
        self,
        image_size: int = 128,
        base_channels: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        ch = int(base_channels)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            layers_per_block=2,
            block_out_channels=(ch, ch * 2, ch * 2),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.unet(x_t, timesteps).sample
