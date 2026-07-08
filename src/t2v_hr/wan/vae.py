from __future__ import annotations

from pathlib import Path

import torch


def load_wan_vae_from_diffusers(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    enable_tiling: bool = True,
    enable_slicing: bool = True,
):
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise ImportError("diffusers with Wan support is required on the experiment server") from exc

    vae = AutoencoderKLWan.from_pretrained(str(model_path), subfolder="vae", torch_dtype=torch_dtype)
    vae = vae.to(device=device, dtype=torch_dtype).eval()
    vae.requires_grad_(False)
    if enable_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if enable_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    return vae


@torch.no_grad()
def encode_video_latents(vae, video: torch.Tensor) -> torch.Tensor:
    posterior = vae.encode(video, return_dict=True).latent_dist
    latents = posterior.mode()
    if hasattr(vae, "config") and hasattr(vae.config, "latents_mean") and hasattr(vae.config, "latents_std"):
        mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, vae.config.z_dim, 1, 1, 1
        )
        std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, vae.config.z_dim, 1, 1, 1
        )
        latents = (latents - mean) / std
    return latents
