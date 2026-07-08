from __future__ import annotations

from pathlib import Path
import re

import torch


def _map_vae_residual(prefix: str, target: str) -> str:
    target = target.replace(f"{prefix}.residual.0.", f"{prefix}.norm1.")
    target = target.replace(f"{prefix}.residual.2.", f"{prefix}.conv1.")
    target = target.replace(f"{prefix}.residual.3.", f"{prefix}.norm2.")
    target = target.replace(f"{prefix}.residual.6.", f"{prefix}.conv2.")
    target = target.replace(f"{prefix}.shortcut.", f"{prefix}.conv_shortcut.")
    return target


def convert_legacy_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state_dict.items():
        new_key = key
        new_key = new_key.replace("encoder.conv1.", "encoder.conv_in.")
        new_key = new_key.replace("encoder.head.0.", "encoder.norm_out.")
        new_key = new_key.replace("encoder.head.2.", "encoder.conv_out.")
        new_key = new_key.replace("decoder.conv1.", "decoder.conv_in.")
        new_key = new_key.replace("decoder.head.0.", "decoder.norm_out.")
        new_key = new_key.replace("decoder.head.2.", "decoder.conv_out.")
        if new_key.startswith("conv1."):
            new_key = new_key.replace("conv1.", "quant_conv.", 1)
        if new_key.startswith("conv2."):
            new_key = new_key.replace("conv2.", "post_quant_conv.", 1)

        match = re.match(r"encoder\.downsamples\.(\d+)\.(.*)", new_key)
        if match:
            idx, rest = int(match.group(1)), match.group(2)
            new_key = f"encoder.down_blocks.{idx}.{rest}"
            new_key = _map_vae_residual(f"encoder.down_blocks.{idx}", new_key)

        match = re.match(r"encoder\.middle\.(\d+)\.(.*)", new_key)
        if match:
            idx, rest = int(match.group(1)), match.group(2)
            if idx == 1:
                new_key = f"encoder.mid_block.attentions.0.{rest}"
            else:
                res_idx = 0 if idx == 0 else 1
                new_key = f"encoder.mid_block.resnets.{res_idx}.{rest}"
                new_key = _map_vae_residual(f"encoder.mid_block.resnets.{res_idx}", new_key)

        match = re.match(r"decoder\.middle\.(\d+)\.(.*)", new_key)
        if match:
            idx, rest = int(match.group(1)), match.group(2)
            if idx == 1:
                new_key = f"decoder.mid_block.attentions.0.{rest}"
            else:
                res_idx = 0 if idx == 0 else 1
                new_key = f"decoder.mid_block.resnets.{res_idx}.{rest}"
                new_key = _map_vae_residual(f"decoder.mid_block.resnets.{res_idx}", new_key)

        match = re.match(r"decoder\.upsamples\.(\d+)\.(.*)", new_key)
        if match:
            idx, rest = int(match.group(1)), match.group(2)
            if idx in {3, 7, 11}:
                block_idx = {3: 0, 7: 1, 11: 2}[idx]
                new_key = f"decoder.up_blocks.{block_idx}.upsamplers.0.{rest}"
            else:
                if idx < 3:
                    block_idx, res_idx = 0, idx
                elif idx < 7:
                    block_idx, res_idx = 1, idx - 4
                elif idx < 11:
                    block_idx, res_idx = 2, idx - 8
                else:
                    block_idx, res_idx = 3, idx - 12
                new_key = f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.{rest}"
                new_key = _map_vae_residual(f"decoder.up_blocks.{block_idx}.resnets.{res_idx}", new_key)
        converted[new_key] = value
    return converted


def load_wan_vae(
    model_path: str | Path,
    *,
    fmt: str = "diffusers",
    vae_weight: str = "Wan2.1_VAE.pth",
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    enable_tiling: bool = True,
    enable_slicing: bool = True,
):
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise ImportError("diffusers with Wan support is required on the experiment server") from exc

    model_path = Path(model_path)
    if fmt == "diffusers":
        vae = AutoencoderKLWan.from_pretrained(str(model_path), subfolder="vae", torch_dtype=torch_dtype)
    elif fmt == "legacy":
        vae = AutoencoderKLWan()
        state = torch.load(model_path / vae_weight, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = vae.load_state_dict(convert_legacy_vae_state_dict(state), strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Wan VAE legacy load failed: missing={len(missing)} unexpected={len(unexpected)} "
                f"missing_head={missing[:8]} unexpected_head={unexpected[:8]}"
            )
    else:
        raise ValueError(f"Unsupported Wan VAE format: {fmt}")
    vae = vae.to(device=device, dtype=torch_dtype).eval()
    vae.requires_grad_(False)
    if enable_tiling and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    if enable_slicing and hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    return vae


def load_wan_vae_from_diffusers(*args, **kwargs):
    return load_wan_vae(*args, fmt="diffusers", **kwargs)


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
