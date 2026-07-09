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


def spatial_blur(x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x_2d = F.avg_pool2d(x_2d, kernel_size=3, stride=1, padding=1)
    return x_2d.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def spatial_high_frequency(x: torch.Tensor) -> torch.Tensor:
    return x - spatial_blur(x)


def spatial_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_h = x[..., 1:, :] - x[..., :-1, :]
    grad_w = x[..., :, 1:] - x[..., :, :-1]
    return grad_h, grad_w


def channel_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dims = (0, 2, 3, 4)
    mean = x.mean(dim=dims)
    std = x.float().std(dim=dims).to(dtype=x.dtype)
    return mean, std


def video_lsr_loss(pred: torch.Tensor, target: torch.Tensor, lr: torch.Tensor, weights: dict) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    losses["latent_l1"] = F.l1_loss(pred, target)
    losses["temporal_l1"] = F.l1_loss(temporal_difference(pred), temporal_difference(target))
    losses["downsample_l1"] = F.l1_loss(downsample_like(pred, lr), lr)
    losses["highfreq_l1"] = F.l1_loss(spatial_high_frequency(pred), spatial_high_frequency(target))
    pred_gh, pred_gw = spatial_gradients(pred)
    target_gh, target_gw = spatial_gradients(target)
    losses["spatial_grad_l1"] = 0.5 * (F.l1_loss(pred_gh, target_gh) + F.l1_loss(pred_gw, target_gw))
    pred_mean, pred_std = channel_moments(pred)
    target_mean, target_std = channel_moments(target)
    losses["channel_stats_l1"] = F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
    total = pred.new_tensor(0.0)
    for key, value in losses.items():
        total = total + float(weights.get(key, 0.0)) * value
    losses["total"] = total
    return losses
