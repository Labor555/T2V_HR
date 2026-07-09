from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_difference(x: torch.Tensor, lag: int = 1) -> torch.Tensor:
    lag = int(lag)
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")
    if x.shape[2] <= lag:
        return torch.zeros_like(x[:, :, :0])
    return x[:, :, lag:] - x[:, :, :-lag]


def temporal_second_difference(x: torch.Tensor) -> torch.Tensor:
    if x.shape[2] < 3:
        return torch.zeros_like(x[:, :, :0])
    return x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]


def downsample_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    target_t, target_h, target_w = target.shape[2:]
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = F.interpolate(x_2d, size=(target_h, target_w), mode="area")
    out = out.reshape(b, t, c, target_h, target_w).permute(0, 2, 1, 3, 4)
    if target_t != t:
        out = out[:, :, :target_t]
    return out


def upsample_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = x.shape
    target_t, target_h, target_w = target.shape[2:]
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    out = F.interpolate(x_2d, size=(target_h, target_w), mode="bilinear", align_corners=False)
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


def multiscale_spatial_high_frequency(x: torch.Tensor, scales: tuple[int, ...] = (2, 4)) -> list[torch.Tensor]:
    b, c, t, h, w = x.shape
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    bands: list[torch.Tensor] = []
    for scale in scales:
        if h < scale or w < scale:
            continue
        low = F.avg_pool2d(x_2d, kernel_size=scale, stride=scale)
        up = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        band = (x_2d - up).reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        bands.append(band)
    return bands


def spatial_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_h = x[..., 1:, :] - x[..., :-1, :]
    grad_w = x[..., :, 1:] - x[..., :, :-1]
    return grad_h, grad_w


def spatial_laplacian(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] < 3 or x.shape[-1] < 3:
        return torch.zeros_like(x[..., :0, :0])
    center = x[..., 1:-1, 1:-1]
    return (
        x[..., :-2, 1:-1]
        + x[..., 2:, 1:-1]
        + x[..., 1:-1, :-2]
        + x[..., 1:-1, 2:]
        - 4.0 * center
    )


def local_standard_deviation(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    b, c, t, h, w = x.shape
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    pad = kernel_size // 2
    mean = F.avg_pool2d(x_2d, kernel_size=kernel_size, stride=1, padding=pad)
    mean_sq = F.avg_pool2d(x_2d * x_2d, kernel_size=kernel_size, stride=1, padding=pad)
    std = torch.sqrt(torch.clamp(mean_sq - mean * mean, min=0.0) + 1e-6)
    return std.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)


def spatial_fft_amplitude(x: torch.Tensor, max_side: int = 128) -> torch.Tensor:
    b, c, t, h, w = x.shape
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).float()
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        out_h = max(8, int(round(h * scale)))
        out_w = max(8, int(round(w * scale)))
        x_2d = F.interpolate(x_2d, size=(out_h, out_w), mode="area")
    amp = torch.fft.rfft2(x_2d, norm="ortho").abs()
    return torch.log1p(amp)


def channel_moments(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dims = (0, 2, 3, 4)
    mean = x.mean(dim=dims)
    std = x.float().std(dim=dims).to(dtype=x.dtype)
    return mean, std


def motion_tolerant_l1(pred: torch.Tensor, target: torch.Tensor, max_shift: int = 1) -> torch.Tensor:
    max_shift = int(max_shift)
    if max_shift <= 0:
        return F.l1_loss(pred, target)
    best = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = torch.roll(target, shifts=(dy, dx), dims=(-2, -1))
            distance = (pred - shifted).abs()
            best = distance if best is None else torch.minimum(best, distance)
    return best.mean()


def edge_weight_from_target(target: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    edge = spatial_high_frequency(target).detach().abs().mean(dim=1, keepdim=True)
    denom = edge.mean(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return 1.0 + float(strength) * edge / denom


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs() * weight).mean()


def video_lsr_loss(pred: torch.Tensor, target: torch.Tensor, lr: torch.Tensor, weights: dict) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    enabled = {key for key, value in weights.items() if float(value) != 0.0}
    base = None
    pred_residual = None
    target_residual = None
    motion_shift = int(weights.get("motion_max_shift", 1))
    edge_strength = float(weights.get("edge_weight_strength", 1.0))
    if "latent_l1" in enabled:
        losses["latent_l1"] = F.l1_loss(pred, target)
    if enabled.intersection(
        {
            "residual_l1",
            "residual_highfreq_l1",
            "residual_grad_l1",
            "residual_laplacian_l1",
            "residual_temporal_l1",
            "motion_residual_highfreq_l1",
        }
    ):
        base = upsample_like(lr, target)
        pred_residual = pred - base
        target_residual = target - base
    if "temporal_l1" in enabled:
        losses["temporal_l1"] = F.l1_loss(temporal_difference(pred), temporal_difference(target))
    if "motion_temporal_l1" in enabled:
        losses["motion_temporal_l1"] = motion_tolerant_l1(
            temporal_difference(pred),
            temporal_difference(target),
            max_shift=motion_shift,
        )
    if "temporal_lag2_l1" in enabled:
        losses["temporal_lag2_l1"] = F.l1_loss(temporal_difference(pred, lag=2), temporal_difference(target, lag=2))
    if "temporal_lag4_l1" in enabled:
        losses["temporal_lag4_l1"] = F.l1_loss(temporal_difference(pred, lag=4), temporal_difference(target, lag=4))
    if "temporal_accel_l1" in enabled:
        losses["temporal_accel_l1"] = F.l1_loss(temporal_second_difference(pred), temporal_second_difference(target))
    if "temporal_highfreq_l1" in enabled:
        losses["temporal_highfreq_l1"] = F.l1_loss(
            spatial_high_frequency(temporal_difference(pred)),
            spatial_high_frequency(temporal_difference(target)),
        )
    if "motion_temporal_highfreq_l1" in enabled:
        losses["motion_temporal_highfreq_l1"] = motion_tolerant_l1(
            spatial_high_frequency(temporal_difference(pred)),
            spatial_high_frequency(temporal_difference(target)),
            max_shift=motion_shift,
        )
    if "downsample_l1" in enabled:
        losses["downsample_l1"] = F.l1_loss(downsample_like(pred, lr), lr)
    if "highfreq_l1" in enabled:
        losses["highfreq_l1"] = F.l1_loss(spatial_high_frequency(pred), spatial_high_frequency(target))
    if "edge_weighted_highfreq_l1" in enabled:
        weight = edge_weight_from_target(target, strength=edge_strength)
        losses["edge_weighted_highfreq_l1"] = weighted_l1(
            spatial_high_frequency(pred),
            spatial_high_frequency(target),
            weight,
        )
    if "multiscale_highfreq_l1" in enabled:
        pred_bands = multiscale_spatial_high_frequency(pred)
        target_bands = multiscale_spatial_high_frequency(target)
        if pred_bands:
            losses["multiscale_highfreq_l1"] = sum(
                F.l1_loss(pred_band, target_band) for pred_band, target_band in zip(pred_bands, target_bands)
            ) / len(pred_bands)
    if "spatial_grad_l1" in enabled:
        pred_gh, pred_gw = spatial_gradients(pred)
        target_gh, target_gw = spatial_gradients(target)
        losses["spatial_grad_l1"] = 0.5 * (F.l1_loss(pred_gh, target_gh) + F.l1_loss(pred_gw, target_gw))
    if "laplacian_l1" in enabled:
        losses["laplacian_l1"] = F.l1_loss(spatial_laplacian(pred), spatial_laplacian(target))
    if "residual_l1" in enabled:
        assert pred_residual is not None and target_residual is not None
        losses["residual_l1"] = F.l1_loss(pred_residual, target_residual)
    if "residual_highfreq_l1" in enabled:
        assert pred_residual is not None and target_residual is not None
        losses["residual_highfreq_l1"] = F.l1_loss(
            spatial_high_frequency(pred_residual),
            spatial_high_frequency(target_residual),
        )
    if "motion_residual_highfreq_l1" in enabled:
        assert pred_residual is not None and target_residual is not None
        losses["motion_residual_highfreq_l1"] = motion_tolerant_l1(
            spatial_high_frequency(pred_residual),
            spatial_high_frequency(target_residual),
            max_shift=motion_shift,
        )
    if "residual_grad_l1" in enabled:
        assert pred_residual is not None and target_residual is not None
        pred_gh, pred_gw = spatial_gradients(pred_residual)
        target_gh, target_gw = spatial_gradients(target_residual)
        losses["residual_grad_l1"] = 0.5 * (F.l1_loss(pred_gh, target_gh) + F.l1_loss(pred_gw, target_gw))
    if "residual_laplacian_l1" in enabled:
        assert pred_residual is not None and target_residual is not None
        losses["residual_laplacian_l1"] = F.l1_loss(spatial_laplacian(pred_residual), spatial_laplacian(target_residual))
    if "residual_temporal_l1" in enabled:
        assert pred_residual is not None and target_residual is not None
        losses["residual_temporal_l1"] = motion_tolerant_l1(
            temporal_difference(pred_residual),
            temporal_difference(target_residual),
            max_shift=motion_shift,
        )
    if "local_std_l1" in enabled:
        losses["local_std_l1"] = F.l1_loss(local_standard_deviation(pred), local_standard_deviation(target))
    if "fft_amp_l1" in enabled:
        losses["fft_amp_l1"] = F.l1_loss(spatial_fft_amplitude(pred), spatial_fft_amplitude(target))
    if "channel_stats_l1" in enabled:
        pred_mean, pred_std = channel_moments(pred)
        target_mean, target_std = channel_moments(target)
        losses["channel_stats_l1"] = F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
    total = pred.new_tensor(0.0)
    for key, value in losses.items():
        total = total + float(weights.get(key, 0.0)) * value
    losses["total"] = total
    return losses
