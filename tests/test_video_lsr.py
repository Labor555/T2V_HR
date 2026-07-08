from __future__ import annotations

import torch

from t2v_hr.models.losses import video_lsr_loss
from t2v_hr.models.video_lsr import VideoLSR
from t2v_hr.rna.video_rna import add_region_time_noise, build_rna_map


def test_video_lsr_shape_and_loss():
    model = VideoLSR(in_channels=4, hidden_channels=16, num_blocks=2, scale_factor=2)
    lr = torch.randn(2, 4, 3, 8, 10)
    hr = torch.randn(2, 4, 3, 16, 20)
    pred = model(lr)
    assert pred.shape == hr.shape
    losses = video_lsr_loss(pred, hr, lr, {"latent_l1": 1.0, "temporal_l1": 0.1, "downsample_l1": 0.1})
    assert losses["total"].ndim == 0
    assert torch.isfinite(losses["total"])


def test_video_rna_shape():
    latents = torch.randn(1, 4, 5, 16, 20)
    detail = build_rna_map(latents)
    noised = add_region_time_noise(latents, detail)
    assert detail.shape == (1, 1, 5, 16, 20)
    assert noised.shape == latents.shape

