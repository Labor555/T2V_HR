from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_difference(x: torch.Tensor) -> torch.Tensor:
    if x.shape[2] < 2:
        return torch.zeros_like(x[:, :, :0])
    return x[:, :, 1:] - x[:, :, :-1]


def downsample_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    target_t, target_h, target_w = target.shape[2:]
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = F.interpolate(x_2d, size=(target_h, target_w), mode="area")
    out = out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
    if target_t != t:
        out = out[:, :, :target_t]
    return out


def video_lsr_loss(pred: torch.Tensor, target: torch.Tensor, lr: torch.Tensor, weights: dict) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    losses["latent_l1"] = F.l1_loss(pred, target)
    losses["temporal_l1"] = F.l1_loss(temporal_difference(pred), temporal_difference(target))
    losses["downsample_l1"] = F.l1_loss(downsample_like(pred, lr), lr)
    total = pred.new_tensor(0.0)
    for key, value in losses.items():
        total = total + float(weights.get(key, 0.0)) * value
    losses["total"] = total
    return losses

