from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    min_v = x.amin(dim=dims, keepdim=True)
    max_v = x.amax(dim=dims, keepdim=True)
    return (x - min_v) / (max_v - min_v + eps)


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


def build_rna_map(
    latents: torch.Tensor,
    *,
    latent_weight: float = 0.5,
    temporal_weight: float = 0.5,
) -> torch.Tensor:
    score = float(latent_weight) * latent_high_frequency(latents)
    score = score + float(temporal_weight) * temporal_detail(latents)
    return _normalize_map(score)


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

