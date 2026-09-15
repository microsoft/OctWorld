# Camera Adapter for OctWorld Wan FramePack
# Adapted from: octworld/models/framepack/models/feature_encoder.py (SimpleAdapter)
#
# Changes:
# - Default out_dim=5120 to match Wan 14B
# - Added temporal dimension handling for video latents
# - Same architecture: PixelUnshuffle + Conv2d + ResidualBlocks
import torch
import torch.nn as nn


__all__ = ['SimpleAdapter', 'ResidualBlock']


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out


class SimpleAdapter(nn.Module):
    """
    Camera/raymap encoder that produces features aligned with patch-embedded latents.
    
    Architecture:
        Input raymap [B, C_ray, T, H, W]
        -> PixelUnshuffle(downscale_factor) per frame
        -> Conv2d(stride) per frame
        -> ResidualBlocks per frame
        -> Output [B, out_dim, T, H_out, W_out]
    
    The output is directly added to the patch-embedded video latents.
    
    For Wan 14B with patch_size=(1,2,2):
        - VAE: 8x spatial downsample -> latent H/8, W/8
        - Patch embedding: 2x spatial -> H/16, W/16
        - SimpleAdapter with downscale_factor=8, stride=2 -> 16x total = H/16, W/16
        => Matches!
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 5120,
        kernel_size: int = 2,
        stride: int = 2,
        downscale_factor: int = 8,
        num_residual_blocks: int = 1,
        zero_init_output_proj: bool = False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downscale_factor = downscale_factor

        # PixelUnshuffle: reduce spatial by downscale_factor
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=downscale_factor)

        # Conv: reduce spatial by stride (e.g., 2x)
        conv_in_dim = in_dim * downscale_factor * downscale_factor
        self.conv = nn.Conv2d(conv_in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=0)

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(out_dim) for _ in range(num_residual_blocks)]
        )

        # Optional zero-init final 1x1 projection so the adapter outputs exactly
        # zero at init. Useful for safe injection into a pretrained DiT.
        self.proj_out = None
        if zero_init_output_proj:
            self.proj_out = nn.Conv2d(out_dim, out_dim, kernel_size=1)
            nn.init.zeros_(self.proj_out.weight)
            nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] raymap/camera features in pixel space
            
        Returns:
            [B, out_dim, T, H_out, W_out] features matching patch-embedded shape
        """
        B, C, T, H, W = x.shape

        # Process each frame independently
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

        # PixelUnshuffle
        x = self.pixel_unshuffle(x)

        # Conv
        x = self.conv(x)

        # Residual blocks
        x = self.residual_blocks(x)

        if self.proj_out is not None:
            x = self.proj_out(x)

        # Reshape back to video format [B, out_dim, T, H_out, W_out]
        _, C_out, H_out, W_out = x.shape
        x = x.view(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)

        return x
