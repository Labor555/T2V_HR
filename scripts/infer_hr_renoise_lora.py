#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

from t2v_hr.models.losses import spatial_gradients, spatial_high_frequency, temporal_difference
from t2v_hr.models.video_lsr import build_video_lsr
from t2v_hr.rna.video_rna import VideoRNAConfig, apply_video_rna
from t2v_hr.utils.config import ensure_dir, load_config
from t2v_hr.utils.torch_utils import dtype_from_string, seed_everything
from t2v_hr.wan.pipeline import (
    WAN_NEGATIVE_PROMPT,
    decode_wan_latents,
    export_video,
    infer_num_frames_from_latents,
    load_wan_lora_into_transformer,
    load_wan_pipeline_legacy,
    refine_wan_from_latents,
    refine_wan_tiled_from_latents,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr-latent", default="")
    parser.add_argument("--hr-latent", default="")
    parser.add_argument("--lsr-checkpoint", default="")
    parser.add_argument("--lora-checkpoint", default="")
    parser.add_argument("--no-lora", action="store_true", help="Ignore config lora.checkpoint and run base Wan.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--latent-crop-lr", default="", help="Optional center crop T,H,W on LR latent.")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--refine-strength", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--hr-denoise-mode", choices=["dense", "tiled"], default="tiled")
    parser.add_argument("--tile-latent-hw", default="64,64")
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--compare-height", type=int, default=540)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def load_latent(path: str | Path) -> torch.Tensor:
    item = torch.load(path, map_location="cpu")
    z = item["latents"] if isinstance(item, dict) else item
    if z.ndim == 4:
        z = z.unsqueeze(0)
    if z.ndim != 5:
        raise ValueError(f"Expected latent [B,C,T,H,W] or [C,T,H,W], got {tuple(z.shape)} from {path}")
    return z


def load_lsr(checkpoint: str | Path, device: torch.device, dtype: torch.dtype):
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_video_lsr(payload["config"].get("model", {}))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device=device, dtype=dtype).eval()


def parse_hw(value: str) -> tuple[int, int]:
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected H,W")
    return parts[0], parts[1]


def parse_crop(value: str) -> tuple[int, int, int] | None:
    if not value:
        return None
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected T,H,W")
    return parts[0], parts[1], parts[2]


def center_crop_latent(z: torch.Tensor, crop: tuple[int, int, int]) -> torch.Tensor:
    _, _, t, h, w = z.shape
    crop_t, crop_h, crop_w = min(crop[0], t), min(crop[1], h), min(crop[2], w)
    t0 = max(0, (t - crop_t) // 2)
    h0 = max(0, (h - crop_h) // 2)
    w0 = max(0, (w - crop_w) // 2)
    return z[:, :, t0 : t0 + crop_t, h0 : h0 + crop_h, w0 : w0 + crop_w].contiguous()


def crop_hr_like_lr(z_hr: torch.Tensor, lr_crop: tuple[int, int, int], scale_factor: int) -> torch.Tensor:
    _, _, t, h, w = z_hr.shape
    crop_t = min(lr_crop[0], t)
    crop_h = min(lr_crop[1] * scale_factor, h)
    crop_w = min(lr_crop[2] * scale_factor, w)
    t0 = max(0, (t - crop_t) // 2)
    h0 = max(0, (h - crop_h) // 2)
    w0 = max(0, (w - crop_w) // 2)
    return z_hr[:, :, t0 : t0 + crop_t, h0 : h0 + crop_h, w0 : w0 + crop_w].contiguous()


def l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.l1_loss(a.float(), b.float()).item())


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, prefix: str) -> dict[str, float]:
    metrics = {
        f"latent_l1_{prefix}_to_wan": l1(pred, target),
        f"temporal_l1_{prefix}_to_wan": l1(temporal_difference(pred), temporal_difference(target)),
        f"highfreq_l1_{prefix}_to_wan": l1(spatial_high_frequency(pred), spatial_high_frequency(target)),
    }
    pred_gh, pred_gw = spatial_gradients(pred)
    target_gh, target_gw = spatial_gradients(target)
    metrics[f"spatial_grad_l1_{prefix}_to_wan"] = 0.5 * (l1(pred_gh, target_gh) + l1(pred_gw, target_gw))
    return metrics


def _resize_frame(frame: np.ndarray, height: int) -> np.ndarray:
    from PIL import Image

    if frame.shape[0] == height:
        return frame
    width = max(1, int(round(frame.shape[1] * height / frame.shape[0])))
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))


def export_side_by_side(videos: list[list[np.ndarray]], path: Path, *, fps: int, height: int) -> None:
    import imageio.v2 as imageio

    frames = []
    count = min(len(video) for video in videos)
    for i in range(count):
        row = []
        for video in videos:
            frame = video[i]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            row.append(_resize_frame(frame, height))
        frames.append(np.concatenate(row, axis=1))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("generation", {}).get("seed", 1234))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(config.get("model", {}).get("torch_dtype", config.get("precision", "bf16")))
    out_dir = ensure_dir(args.output_dir or config.get("output", {}).get("dir", "outputs/hr_renoise_lora_eval"))

    if not args.lr_latent:
        raise ValueError("--lr-latent is required for this evaluation script.")
    z_lr = load_latent(args.lr_latent).to(device=device, dtype=dtype)
    z_hr_target = load_latent(args.hr_latent).to(device=device, dtype=dtype) if args.hr_latent else None

    lsr_ckpt = args.lsr_checkpoint or config["lsr"]["checkpoint"]
    lsr = load_lsr(lsr_ckpt, device, dtype)
    scale_factor = int(config.get("lsr", {}).get("scale_factor", getattr(lsr, "scale_factor", 1)))
    crop = parse_crop(args.latent_crop_lr)
    if crop is not None:
        z_lr = center_crop_latent(z_lr, crop)
        if z_hr_target is not None:
            z_hr_target = crop_hr_like_lr(z_hr_target, crop, scale_factor)

    with torch.no_grad():
        z_hr_ref = lsr(z_lr)
        rna_cfg = config.get("rna", {})
        if bool(rna_cfg.get("enabled", False)):
            z_hr_ref, detail = apply_video_rna(
                z_hr_ref,
                config=VideoRNAConfig(
                    latent_weight=float(rna_cfg.get("latent_weight", 0.4)),
                    temporal_weight=float(rna_cfg.get("temporal_weight", 0.3)),
                    edge_weight=float(rna_cfg.get("edge_weight", 0.0)),
                    smooth_kernel=int(rna_cfg.get("smooth_kernel", 3)),
                    gamma=float(rna_cfg.get("gamma", 1.0)),
                    base_strength=float(rna_cfg.get("base_strength", 0.02)),
                    detail_strength=float(rna_cfg.get("detail_strength", 0.18)),
                ),
            )
        else:
            detail = torch.zeros_like(z_hr_ref[:, :1])

    gen_cfg = config.get("generation", {})
    pipe = load_wan_pipeline_legacy(
        config["model"]["model_path"],
        torch_dtype=dtype,
        device=device,
        vae_format=str(config["model"].get("format", "legacy")).lower(),
        vae_weight=str(config["model"].get("vae_weight", "Wan2.1_VAE.pth")),
        flow_shift=float(config.get("scheduler", {}).get("flow_shift", config["model"].get("flow_shift", 5.0))),
    )
    lora_ckpt = "" if args.no_lora else (args.lora_checkpoint or config.get("lora", {}).get("checkpoint", ""))
    if lora_ckpt:
        result, payload = load_wan_lora_into_transformer(pipe.transformer, lora_ckpt)
        print(f"loaded_lora={lora_ckpt} step={payload.get('step')} modules={len(result.module_names)}", flush=True)
    else:
        print("loaded_lora=none; running base Wan HR denoise", flush=True)

    refine_strength = float(args.refine_strength or gen_cfg.get("refine_strength", 0.35))
    num_inference_steps = int(args.num_inference_steps or gen_cfg.get("num_inference_steps", 50))
    generator = torch.Generator(device=device).manual_seed(seed)
    common_kwargs = {
        "prompt": gen_cfg.get("prompt", "high"),
        "negative_prompt": gen_cfg.get("negative_prompt", WAN_NEGATIVE_PROMPT),
        "num_inference_steps": num_inference_steps,
        "refine_strength": refine_strength,
        "guidance_scale": float(args.guidance_scale or gen_cfg.get("guidance_scale", 5.0)),
        "max_sequence_length": int(gen_cfg.get("max_sequence_length", 512)),
        "renoise": True,
        "generator": generator,
        "output_type": "latent",
    }
    if args.hr_denoise_mode == "tiled":
        z_hr_denoised = refine_wan_tiled_from_latents(
            pipe,
            z_hr_ref,
            tile_hw=parse_hw(args.tile_latent_hw),
            overlap=int(args.tile_overlap),
            **common_kwargs,
        )
    else:
        height = int(z_hr_ref.shape[-2]) * int(pipe.vae_scale_factor_spatial)
        width = int(z_hr_ref.shape[-1]) * int(pipe.vae_scale_factor_spatial)
        num_frames = infer_num_frames_from_latents(z_hr_ref, temporal_scale=int(pipe.vae_scale_factor_temporal))
        z_hr_denoised = refine_wan_from_latents(
            pipe,
            z_hr_ref,
            height=height,
            width=width,
            num_frames=num_frames,
            **common_kwargs,
        )

    latents_path = out_dir / "hr_renoise_lora_latents.pt"
    payload = {
        "z_lr": z_lr.detach().cpu(),
        "z_hr_ref": z_hr_ref.detach().cpu(),
        "z_hr_denoised": z_hr_denoised.detach().cpu(),
        "detail_map": detail.detach().cpu(),
        "config": config,
    }
    if z_hr_target is not None:
        payload["z_hr_target"] = z_hr_target.detach().cpu()
    torch.save(payload, latents_path)
    print(f"wrote_latents={latents_path}", flush=True)

    metrics: dict[str, float | int | str] = {
        "num_inference_steps": num_inference_steps,
        "refine_strength": refine_strength,
        "hr_denoise_mode": args.hr_denoise_mode,
    }
    if z_hr_target is not None:
        metrics.update(compute_metrics(z_hr_ref.detach().cpu(), z_hr_target.detach().cpu(), "lsrna"))
        metrics.update(compute_metrics(z_hr_denoised.detach().cpu(), z_hr_target.detach().cpu(), "hrdenoise"))
        metrics["delta_highfreq_hrdenoise_minus_lsrna"] = (
            metrics["highfreq_l1_hrdenoise_to_wan"] - metrics["highfreq_l1_lsrna_to_wan"]
        )
        metrics["delta_spatial_grad_hrdenoise_minus_lsrna"] = (
            metrics["spatial_grad_l1_hrdenoise_to_wan"] - metrics["spatial_grad_l1_lsrna_to_wan"]
        )
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True), flush=True)

    if args.skip_video:
        return
    ref_frames = decode_wan_latents(pipe, z_hr_ref, output_type="np")
    denoised_frames = decode_wan_latents(pipe, z_hr_denoised, output_type="np")
    ref_path = export_video(ref_frames, out_dir / "lsrna_coarse.mp4", fps=args.fps)
    denoised_path = export_video(denoised_frames, out_dir / "hr_renoise_lora.mp4", fps=args.fps)
    print(f"wrote_video={ref_path}", flush=True)
    print(f"wrote_video={denoised_path}", flush=True)
    if z_hr_target is not None:
        target_frames = decode_wan_latents(pipe, z_hr_target, output_type="np")
        target_path = export_video(target_frames, out_dir / "wan_vae_target.mp4", fps=args.fps)
        compare_path = out_dir / f"compare_lsrna_hrdenoise_target_{args.compare_height}p.mp4"
        export_side_by_side(
            [ref_frames[0] if isinstance(ref_frames[0], list) else ref_frames, denoised_frames[0] if isinstance(denoised_frames[0], list) else denoised_frames, target_frames[0] if isinstance(target_frames[0], list) else target_frames],
            compare_path,
            fps=args.fps,
            height=int(args.compare_height),
        )
        print(f"wrote_video={target_path}", flush=True)
        print(f"wrote_compare={compare_path}", flush=True)


if __name__ == "__main__":
    main()
