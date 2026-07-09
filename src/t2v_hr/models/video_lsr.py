from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, temporal_kernel: int = 3, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        padding_t = temporal_kernel // 2
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(temporal_kernel, 3, 3), padding=(padding_t, 1, 1))
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(temporal_kernel, 3, 3), padding=(padding_t, 1, 1))

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class VideoLSR(nn.Module):
    """Small temporal latent super-resolution model.

    The model maps Wan-style latent tensors `[B, C, T, H, W]` to
    `[B, C, T, scale*H, scale*W]`.
    """

    def __init__(
        self,
        in_channels: int = 16,
        hidden_channels: int = 128,
        num_blocks: int = 8,
        temporal_kernel: int = 3,
        scale_factor: int = 2,
        residual_learning: bool = True,
        residual_scale: float = 1.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        if scale_factor < 1:
            raise ValueError("scale_factor must be >= 1")
        self.in_channels = int(in_channels)
        self.scale_factor = int(scale_factor)
        self.residual_learning = bool(residual_learning)
        self.residual_scale = float(residual_scale)
        self.proj_in = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[
                ResidualTemporalBlock(
                    hidden_channels,
                    temporal_kernel=temporal_kernel,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(num_blocks)
            ]
        )
        self.proj_out = nn.Conv3d(hidden_channels, in_channels * scale_factor * scale_factor, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        b, c, t, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        base = F.interpolate(x, scale_factor=(1, self.scale_factor, self.scale_factor), mode="trilinear", align_corners=False)
        x = self.proj_out(self.blocks(self.proj_in(x)))
        scale = self.scale_factor
        x = x.view(b, c, scale, scale, t, h, w)
        x = x.permute(0, 1, 4, 5, 2, 6, 3).contiguous()
        residual = x.view(b, c, t, h * scale, w * scale)
        if self.residual_learning:
            return base + self.residual_scale * residual
        return residual


def build_video_lsr(config: dict) -> VideoLSR:
    return VideoLSR(
        in_channels=int(config.get("in_channels", 16)),
        hidden_channels=int(config.get("hidden_channels", 128)),
        num_blocks=int(config.get("num_blocks", 8)),
        temporal_kernel=int(config.get("temporal_kernel", 3)),
        scale_factor=int(config.get("scale_factor", 2)),
        residual_learning=bool(config.get("residual_learning", True)),
        residual_scale=float(config.get("residual_scale", 1.0)),
        use_checkpoint=bool(config.get("use_checkpoint", False)),
    )
