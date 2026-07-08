from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VideoRNAConfig:
    latent_weight: float = 0.4
    temporal_weight: float = 0.3
    edge_weight: float = 0.0
    smooth_kernel: int = 3
    gamma: float = 1.0
    base_strength: float = 0.02
    detail_strength: float = 0.18


def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    min_v = x.amin(dim=dims, keepdim=True)
    max_v = x.amax(dim=dims, keepdim=True)
    return (x - min_v) / (max_v - min_v + eps)


def _resize_map(score: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    b, c, t, h, w = score.shape
    flat = score.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = F.interpolate(flat, size=size, mode="bilinear", align_corners=False)
    return flat.reshape(b, t, c, *size).permute(0, 2, 1, 3, 4)


def _smooth_map(score: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return score
    if kernel_size % 2 == 0:
        kernel_size += 1
    b, c, t, h, w = score.shape
    flat = score.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = F.avg_pool2d(flat, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def latent_high_frequency(latents: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = latents.shape
    flat = latents.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    blur = F.avg_pool2d(flat, kernel_size=5, stride=1, padding=2)
    hp = (flat - blur).abs().mean(dim=1, keepdim=True)
    hp = hp.reshape(b, t, 1, h, w).permute(0, 2, 1, 3, 4)
    return _normalize_map(hp)


def temporal_detail(latents: torch.Tensor) -> torch.Tensor:
    if latents.shape[2] < 2:
        return torch.zeros(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4], device=latents.device)
    diff = (latents[:, :, 1:] - latents[:, :, :-1]).abs().mean(dim=1, keepdim=True)
    diff = F.pad(diff, (0, 0, 0, 0, 1, 0), mode="replicate")
    return _normalize_map(diff)


def rgb_sobel_edges(video: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    """Sobel edge map from decoded video `[B,3,T,H,W]` in any numeric range."""

    if video.ndim != 5:
        raise ValueError(f"Expected decoded video [B,3,T,H,W], got {tuple(video.shape)}")
    b, c, t, h, w = video.shape
    gray = video.mean(dim=1, keepdim=True)
    flat = gray.permute(0, 2, 1, 3, 4).reshape(b * t, 1, h, w).float()
    sobel_x = flat.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sobel_y = flat.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    gx = F.conv2d(flat, sobel_x, padding=1)
    gy = F.conv2d(flat, sobel_y, padding=1)
    edges = torch.sqrt(gx.square() + gy.square() + 1e-8)
    edges = F.interpolate(edges, size=output_size, mode="bilinear", align_corners=False)
    edges = edges.reshape(b, t, 1, *output_size).permute(0, 2, 1, 3, 4)
    return _normalize_map(edges)


def build_rna_map(
    latents: torch.Tensor,
    *,
    decoded_video: torch.Tensor | None = None,
    latent_weight: float = 0.4,
    temporal_weight: float = 0.3,
    edge_weight: float = 0.0,
    smooth_kernel: int = 3,
    gamma: float = 1.0,
) -> torch.Tensor:
    score = float(latent_weight) * latent_high_frequency(latents)
    score = score + float(temporal_weight) * temporal_detail(latents)
    if decoded_video is not None and edge_weight > 0:
        score = score + float(edge_weight) * rgb_sobel_edges(decoded_video, latents.shape[-2:])
    score = _normalize_map(score)
    score = _smooth_map(score, int(smooth_kernel))
    score = _normalize_map(score).clamp(0, 1)
    if gamma != 1.0:
        score = score.pow(float(gamma))
        score = _normalize_map(score).clamp(0, 1)
    return score


def add_region_time_noise(
    latents: torch.Tensor,
    detail_map: torch.Tensor,
    *,
    base_strength: float = 0.02,
    detail_strength: float = 0.18,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    strength = float(base_strength) + float(detail_strength) * detail_map.to(latents.dtype)
    return latents + strength * noise


def apply_video_rna(
    latents: torch.Tensor,
    *,
    decoded_video: torch.Tensor | None = None,
    config: VideoRNAConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = config or VideoRNAConfig()
    detail = build_rna_map(
        latents.float(),
        decoded_video=decoded_video,
        latent_weight=cfg.latent_weight,
        temporal_weight=cfg.temporal_weight,
        edge_weight=cfg.edge_weight,
        smooth_kernel=cfg.smooth_kernel,
        gamma=cfg.gamma,
    ).to(device=latents.device, dtype=latents.dtype)
    noised = add_region_time_noise(
        latents,
        detail,
        base_strength=cfg.base_strength,
        detail_strength=cfg.detail_strength,
        generator=generator,
    )
    return noised, detail
