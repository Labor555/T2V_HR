from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from t2v_hr.wan.vae import load_wan_vae


WAN_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
)


def _legacy_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    dim = int(config.get("dim", 1536))
    heads = int(config.get("num_heads", 12))
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": heads,
        "attention_head_dim": dim // heads,
        "in_channels": int(config.get("in_dim", 16)),
        "out_channels": int(config.get("out_dim", 16)),
        "text_dim": 4096,
        "freq_dim": int(config.get("freq_dim", 256)),
        "ffn_dim": int(config.get("ffn_dim", 8960)),
        "num_layers": int(config.get("num_layers", 30)),
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": float(config.get("eps", 1e-6)),
    }


def load_wan_transformer_legacy(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
):
    import json

    from diffusers import WanTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
    from safetensors.torch import load_file

    model_path = Path(model_path)
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    transformer = WanTransformer3DModel(**_legacy_transformer_config(config))
    state = load_file(str(model_path / "diffusion_pytorch_model.safetensors"), device="cpu")
    state = convert_wan_transformer_to_diffusers(dict(state))
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Wan transformer legacy load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_head={missing[:8]} unexpected_head={unexpected[:8]}"
        )
    transformer = transformer.to(device=device, dtype=torch_dtype).eval()
    transformer.requires_grad_(False)
    return transformer


def _convert_wan_t5_state_dict(state: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int, int]:
    import re

    blocks = sorted({int(re.match(r"blocks\.(\d+)\.", key).group(1)) for key in state if key.startswith("blocks.")})
    if not blocks:
        raise ValueError("No `blocks.*` keys found in legacy Wan T5 checkpoint.")
    num_layers = max(blocks) + 1
    vocab_size, d_model = state["token_embedding.weight"].shape
    converted: dict[str, torch.Tensor] = {
        "shared.weight": state["token_embedding.weight"],
        "encoder.embed_tokens.weight": state["token_embedding.weight"],
        "encoder.final_layer_norm.weight": state["norm.weight"],
    }
    for i in range(num_layers):
        src = f"blocks.{i}"
        dst = f"encoder.block.{i}"
        converted[f"{dst}.layer.0.layer_norm.weight"] = state[f"{src}.norm1.weight"]
        converted[f"{dst}.layer.0.SelfAttention.q.weight"] = state[f"{src}.attn.q.weight"]
        converted[f"{dst}.layer.0.SelfAttention.k.weight"] = state[f"{src}.attn.k.weight"]
        converted[f"{dst}.layer.0.SelfAttention.v.weight"] = state[f"{src}.attn.v.weight"]
        converted[f"{dst}.layer.0.SelfAttention.o.weight"] = state[f"{src}.attn.o.weight"]
        converted[f"{dst}.layer.0.SelfAttention.relative_attention_bias.weight"] = state[
            f"{src}.pos_embedding.embedding.weight"
        ]
        converted[f"{dst}.layer.1.layer_norm.weight"] = state[f"{src}.norm2.weight"]
        converted[f"{dst}.layer.1.DenseReluDense.wi_0.weight"] = state[f"{src}.ffn.gate.0.weight"]
        converted[f"{dst}.layer.1.DenseReluDense.wi_1.weight"] = state[f"{src}.ffn.fc1.weight"]
        converted[f"{dst}.layer.1.DenseReluDense.wo.weight"] = state[f"{src}.ffn.fc2.weight"]
    return converted, num_layers, vocab_size


def load_wan_text_encoder_legacy(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
):
    from transformers import UMT5Config, UMT5EncoderModel

    model_path = Path(model_path)
    state = torch.load(model_path / "models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu")
    state, num_layers, vocab_size = _convert_wan_t5_state_dict(state)
    config = UMT5Config(
        vocab_size=vocab_size,
        d_model=4096,
        d_ff=10240,
        num_layers=num_layers,
        num_decoder_layers=0,
        num_heads=64,
        d_kv=64,
        is_encoder_decoder=False,
        feed_forward_proj="gated-gelu",
        layer_norm_epsilon=1e-6,
        dropout_rate=0.0,
    )
    text_encoder = UMT5EncoderModel(config)
    missing, unexpected = text_encoder.load_state_dict(state, strict=False)
    allowed_missing = {"encoder.embed_tokens.weight"}
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            "Wan T5 legacy load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_head={missing[:8]} unexpected_head={unexpected[:8]}"
        )
    text_encoder = text_encoder.to(device=device, dtype=torch_dtype).eval()
    text_encoder.requires_grad_(False)
    return text_encoder


def load_wan_pipeline_legacy(
    model_path: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
    vae_format: str = "legacy",
    vae_weight: str = "Wan2.1_VAE.pth",
    flow_shift: float = 5.0,
    load_text_encoder: bool = True,
    load_transformer: bool = True,
    load_vae: bool = True,
):
    from diffusers import WanPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer

    model_path = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "google" / "umt5-xxl"), local_files_only=True)
    text_encoder = (
        load_wan_text_encoder_legacy(model_path, torch_dtype=torch_dtype, device=device) if load_text_encoder else None
    )
    transformer = (
        load_wan_transformer_legacy(model_path, torch_dtype=torch_dtype, device=device) if load_transformer else None
    )
    vae = (
        load_wan_vae(
            model_path,
            fmt=vae_format,
            vae_weight=vae_weight,
            torch_dtype=torch_dtype,
            device=device,
            enable_tiling=True,
            enable_slicing=True,
        )
        if load_vae
        else None
    )
    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    pipe = WanPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
    )
    pipe = pipe.to(device)
    return pipe


def infer_num_frames_from_latents(latents: torch.Tensor, temporal_scale: int = 4) -> int:
    latent_frames = int(latents.shape[2])
    return (latent_frames - 1) * temporal_scale + 1


def denormalize_wan_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents.to(dtype=vae.dtype, device=latents.device)
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    return latents * latents_std + latents_mean


@torch.no_grad()
def decode_wan_latents(pipe, latents: torch.Tensor, *, output_type: str = "np"):
    latents = denormalize_wan_latents(pipe.vae, latents.to(pipe._execution_device))
    video = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


def refine_wan_from_latents(
    pipe,
    latents: torch.Tensor,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int = 50,
    refine_strength: float = 0.35,
    guidance_scale: float = 5.0,
    max_sequence_length: int = 512,
    attention_kwargs: dict[str, Any] | None = None,
    output_type: str = "np",
):
    if not 0 < refine_strength <= 1:
        raise ValueError("refine_strength must be in (0, 1].")
    device = pipe._execution_device
    latents = latents.to(device=device, dtype=torch.float32)
    pipe.check_inputs(prompt, negative_prompt, height, width, None, None, ["latents"], None)
    pipe._guidance_scale = guidance_scale
    pipe._guidance_scale_2 = None
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False

    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt or WAN_NEGATIVE_PROMPT,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    latents = denoise_wan_latents(
        pipe,
        latents,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=num_inference_steps,
        refine_strength=refine_strength,
        guidance_scale=guidance_scale,
        attention_kwargs=attention_kwargs,
    )
    if output_type == "latent":
        return latents
    return decode_wan_latents(pipe, latents, output_type=output_type)


def denoise_wan_latents(
    pipe,
    latents: torch.Tensor,
    *,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None = None,
    num_inference_steps: int = 50,
    refine_strength: float = 0.35,
    guidance_scale: float = 5.0,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    if not 0 < refine_strength <= 1:
        raise ValueError("refine_strength must be in (0, 1].")
    device = pipe._execution_device
    latents = latents.to(device=device, dtype=torch.float32)
    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)
    if negative_prompt_embeds is not None:
        negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=transformer_dtype)
    do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    start_index = max(0, min(len(timesteps) - 1, int(round((1.0 - refine_strength) * len(timesteps)))))
    timesteps = timesteps[start_index:]
    pipe.scheduler.set_begin_index(start_index)
    pipe._num_timesteps = len(timesteps)
    mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

    with pipe.progress_bar(total=len(timesteps)) as progress_bar:
        for t in timesteps:
            pipe._current_timestep = t
            latent_model_input = latents.to(transformer_dtype)
            if pipe.config.expand_timesteps:
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                timestep = t.expand(latents.shape[0])
            with pipe.transformer.cache_context("cond"):
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
            if do_cfg:
                with pipe.transformer.cache_context("uncond"):
                    noise_uncond = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            progress_bar.update()

    pipe._current_timestep = None
    return latents


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0:
        raise ValueError("tile size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if tile >= length:
        return [0]
    stride = max(1, tile - overlap)
    starts = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tile_blend_weight(
    tile_h: int,
    tile_w: int,
    *,
    y0: int,
    x0: int,
    full_h: int,
    full_w: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    wy = torch.ones(tile_h, device=device, dtype=dtype)
    wx = torch.ones(tile_w, device=device, dtype=dtype)
    if overlap > 0:
        oy = min(overlap, tile_h)
        ox = min(overlap, tile_w)
        if y0 > 0 and oy > 1:
            wy[:oy] = torch.linspace(0.05, 1.0, oy, device=device, dtype=dtype)
        if y0 + tile_h < full_h and oy > 1:
            wy[-oy:] = torch.linspace(1.0, 0.05, oy, device=device, dtype=dtype)
        if x0 > 0 and ox > 1:
            wx[:ox] = torch.linspace(0.05, 1.0, ox, device=device, dtype=dtype)
        if x0 + tile_w < full_w and ox > 1:
            wx[-ox:] = torch.linspace(1.0, 0.05, ox, device=device, dtype=dtype)
    return wy.view(1, 1, 1, tile_h, 1) * wx.view(1, 1, 1, 1, tile_w)


def refine_wan_tiled_from_latents(
    pipe,
    latents: torch.Tensor,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    num_inference_steps: int = 50,
    refine_strength: float = 0.35,
    guidance_scale: float = 5.0,
    max_sequence_length: int = 512,
    tile_hw: tuple[int, int] = (64, 64),
    overlap: int = 16,
    attention_kwargs: dict[str, Any] | None = None,
    output_type: str = "np",
):
    device = pipe._execution_device
    latents = latents.to(device=device, dtype=torch.float32)
    _, _, _, full_h, full_w = latents.shape
    tile_h = min(int(tile_hw[0]), full_h)
    tile_w = min(int(tile_hw[1]), full_w)
    y_starts = _tile_starts(full_h, tile_h, overlap)
    x_starts = _tile_starts(full_w, tile_w, overlap)

    pipe._guidance_scale = guidance_scale
    pipe._guidance_scale_2 = None
    pipe._attention_kwargs = attention_kwargs
    pipe._current_timestep = None
    pipe._interrupt = False
    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt or WAN_NEGATIVE_PROMPT,
        do_classifier_free_guidance=do_cfg,
        num_videos_per_prompt=1,
        max_sequence_length=max_sequence_length,
        device=device,
    )

    accum = torch.zeros_like(latents)
    weights = torch.zeros((1, 1, 1, full_h, full_w), device=device, dtype=latents.dtype)
    total = len(y_starts) * len(x_starts)
    tile_index = 0
    for y0 in y_starts:
        for x0 in x_starts:
            tile_index += 1
            y1, x1 = y0 + tile_h, x0 + tile_w
            print(f"refine_tile={tile_index}/{total} y={y0}:{y1} x={x0}:{x1}", flush=True)
            tile = latents[:, :, :, y0:y1, x0:x1].contiguous()
            refined = denoise_wan_latents(
                pipe,
                tile,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                refine_strength=refine_strength,
                guidance_scale=guidance_scale,
                attention_kwargs=attention_kwargs,
            )
            weight = _tile_blend_weight(
                tile_h,
                tile_w,
                y0=y0,
                x0=x0,
                full_h=full_h,
                full_w=full_w,
                overlap=overlap,
                device=device,
                dtype=latents.dtype,
            )
            accum[:, :, :, y0:y1, x0:x1] += refined * weight
            weights[:, :, :, y0:y1, x0:x1] += weight
            del tile, refined, weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    refined_latents = accum / weights.clamp_min(1e-6)
    if output_type == "latent":
        return refined_latents
    return decode_wan_latents(pipe, refined_latents, output_type=output_type)


def export_video(frames, path: str | Path, *, fps: int = 16) -> Path:
    import imageio.v2 as imageio
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    video = frames
    if isinstance(frames, list) and frames:
        if isinstance(frames[0], list):
            video = frames[0]
        elif hasattr(frames[0], "__array__") and np.asarray(frames[0]).ndim == 4:
            video = frames[0]
    if hasattr(video, "detach"):
        video = video.detach().cpu().numpy()
    if hasattr(video, "__array__"):
        video = np.asarray(video)
        if video.ndim == 5:
            video = video[0]
    arrays = []
    for frame in video:
        if hasattr(frame, "__array__"):
            arr = np.asarray(frame)
        else:
            arr = frame
        if arr.dtype != np.uint8:
            arr = (arr.clip(0, 1) * 255).astype(np.uint8)
        arrays.append(arr)
    imageio.mimsave(path, arrays, fps=fps, macro_block_size=1)
    return path
