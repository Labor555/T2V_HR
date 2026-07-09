#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from t2v_hr.models.video_lsr import build_video_lsr
from t2v_hr.rna.video_rna import VideoRNAConfig, apply_video_rna
from t2v_hr.utils.config import ensure_dir, load_config
from t2v_hr.utils.torch_utils import dtype_from_string, seed_everything
from t2v_hr.wan.pipeline import (
    WAN_NEGATIVE_PROMPT,
    decode_wan_latents,
    export_video,
    infer_num_frames_from_latents,
    load_wan_pipeline_legacy,
    refine_wan_from_latents,
    refine_wan_tiled_from_latents,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr-latent", default="", help="Optional cached low-res latent tensor.")
    parser.add_argument("--hr-latent", default="", help="Optional cached high-res latent tensor for target comparison.")
    parser.add_argument("--lsr-checkpoint", default="", help="Override config lsr.checkpoint.")
    parser.add_argument(
        "--mode",
        choices=["lsrna", "full"],
        default="full",
        help="`lsrna` only writes Video-LSR/RNA latents; `full` also runs Wan HR denoise/refine and exports video.",
    )
    parser.add_argument(
        "--latent-crop-lr",
        default="",
        help="Optional center crop on LR latent as T,H,W, e.g. 5,32,32 for fast HR refine tests.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--refine-strength", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--output-dir", default="", help="Optional output directory override.")
    parser.add_argument(
        "--hr-denoise-mode",
        choices=["dense", "tiled"],
        default="tiled",
        help="Use tiled HR denoise for full 4K latent grids; dense is only practical for small crops.",
    )
    parser.add_argument("--tile-latent-hw", default="64,64", help="HR latent tile size H,W for tiled denoise.")
    parser.add_argument("--tile-overlap", type=int, default=16, help="HR latent overlap for tiled denoise.")
    parser.add_argument("--save-intermediates", action="store_true", help="Decode and save target/LSR reference videos.")
    return parser.parse_args()


def load_lsr(checkpoint: str | Path, device: torch.device, dtype: torch.dtype):
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_video_lsr(payload["config"].get("model", {}))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device=device, dtype=dtype).eval()


def load_latent(path: str | Path) -> torch.Tensor:
    item = torch.load(path, map_location="cpu")
    z = item["latents"] if isinstance(item, dict) else item
    if z.ndim == 4:
        z = z.unsqueeze(0)
    if z.ndim != 5:
        raise ValueError(f"Expected latent [B,C,T,H,W] or [C,T,H,W], got {tuple(z.shape)} from {path}")
    return z


def parse_latent_crop(value: str) -> tuple[int, int, int] | None:
    if not value:
        return None
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--latent-crop-lr must be formatted as T,H,W")
    return parts[0], parts[1], parts[2]


def parse_hw(value: str) -> tuple[int, int]:
    parts = [int(part) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("Expected H,W")
    return parts[0], parts[1]


def center_crop_latent(z: torch.Tensor, crop: tuple[int, int, int]) -> torch.Tensor:
    _, _, t, h, w = z.shape
    crop_t, crop_h, crop_w = crop
    crop_t = min(crop_t, t)
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)
    t0 = max(0, (t - crop_t) // 2)
    h0 = max(0, (h - crop_h) // 2)
    w0 = max(0, (w - crop_w) // 2)
    return z[:, :, t0 : t0 + crop_t, h0 : h0 + crop_h, w0 : w0 + crop_w].contiguous()


def crop_hr_like_lr(z_hr: torch.Tensor, lr_crop: tuple[int, int, int], scale_factor: int) -> torch.Tensor:
    _, _, t, h, w = z_hr.shape
    crop_t, crop_h, crop_w = lr_crop
    crop_t = min(crop_t, t)
    crop_h = min(crop_h * scale_factor, h)
    crop_w = min(crop_w * scale_factor, w)
    t0 = max(0, (t - crop_t) // 2)
    h0 = max(0, (h - crop_h) // 2)
    w0 = max(0, (w - crop_w) // 2)
    return z_hr[:, :, t0 : t0 + crop_t, h0 : h0 + crop_h, w0 : w0 + crop_w].contiguous()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("generation", {}).get("seed", 1234)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(config.get("model", {}).get("torch_dtype", "bf16"))
    out_dir = ensure_dir(args.output_dir or config.get("output", {}).get("dir", "outputs/infer_lsrna"))

    lsr = load_lsr(args.lsr_checkpoint or config["lsr"]["checkpoint"], device, dtype)
    scale_factor = int(config.get("lsr", {}).get("scale_factor", getattr(lsr, "scale_factor", 1)))

    if args.lr_latent:
        z_lr = load_latent(args.lr_latent)
        z_lr = z_lr.to(device=device, dtype=dtype)
    else:
        pipe = load_wan_pipeline_legacy(
            config["model"]["model_path"],
            torch_dtype=dtype,
            device=device,
            vae_format=str(config["model"].get("format", "legacy")).lower(),
            vae_weight=str(config["model"].get("vae_weight", "Wan2.1_VAE.pth")),
            flow_shift=float(config.get("scheduler", {}).get("flow_shift", 5.0)),
            load_vae=bool(config.get("generation", {}).get("decode_lowres", False)),
        )
        gen_cfg = config.get("generation", {})
        generator = torch.Generator(device=device).manual_seed(int(gen_cfg.get("seed", 1234)))
        output = pipe(
            prompt=gen_cfg.get("prompt", "high quality video"),
            negative_prompt=gen_cfg.get("negative_prompt", WAN_NEGATIVE_PROMPT),
            height=int(gen_cfg.get("low_resolution", [720, 1280])[0]),
            width=int(gen_cfg.get("low_resolution", [720, 1280])[1]),
            num_frames=int(gen_cfg.get("num_frames", 81)),
            num_inference_steps=int(args.num_inference_steps or gen_cfg.get("num_inference_steps", 50)),
            guidance_scale=float(args.guidance_scale or gen_cfg.get("guidance_scale", 5.0)),
            generator=generator,
            output_type="latent",
        )
        z_lr = output.frames if isinstance(output.frames, torch.Tensor) else output.frames[0]
        if z_lr.ndim == 4:
            z_lr = z_lr.unsqueeze(0)
        z_lr = z_lr.to(device=device, dtype=dtype)

    crop = parse_latent_crop(args.latent_crop_lr)
    z_hr_target = None
    if crop is not None:
        z_lr = center_crop_latent(z_lr, crop)
    if args.hr_latent:
        z_hr_target = load_latent(args.hr_latent).to(device=device, dtype=dtype)
        if crop is not None:
            z_hr_target = crop_hr_like_lr(z_hr_target, crop, scale_factor)

    with torch.no_grad():
        z_ref = lsr(z_lr)
        rna_cfg = config.get("rna", {})
        z_noisy, detail = apply_video_rna(
            z_ref,
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

    payload = {"z_lr": z_lr.cpu(), "z_hr_ref": z_ref.cpu(), "detail_map": detail.cpu(), "z_noisy": z_noisy.cpu()}
    if z_hr_target is not None:
        payload["z_hr_target"] = z_hr_target.cpu()
    torch.save(payload, out_dir / "lsrna_latents.pt")
    print(f"Wrote Video-LSRNA latents to {out_dir / 'lsrna_latents.pt'}")

    if args.mode == "lsrna":
        print("Skipped Wan HR denoise/refine because --mode=lsrna.")
        return

    gen_cfg = config.get("generation", {})
    pipe = load_wan_pipeline_legacy(
        config["model"]["model_path"],
        torch_dtype=dtype,
        device=device,
        vae_format=str(config["model"].get("format", "legacy")).lower(),
        vae_weight=str(config["model"].get("vae_weight", "Wan2.1_VAE.pth")),
        flow_shift=float(config.get("scheduler", {}).get("flow_shift", 5.0)),
    )
    height = int(z_noisy.shape[-2]) * int(pipe.vae_scale_factor_spatial)
    width = int(z_noisy.shape[-1]) * int(pipe.vae_scale_factor_spatial)
    num_frames = infer_num_frames_from_latents(z_noisy, temporal_scale=int(pipe.vae_scale_factor_temporal))
    refine_kwargs = {
        "prompt": gen_cfg.get("prompt", "high quality video"),
        "negative_prompt": gen_cfg.get("negative_prompt", WAN_NEGATIVE_PROMPT),
        "num_inference_steps": int(args.num_inference_steps or gen_cfg.get("num_inference_steps", 50)),
        "refine_strength": float(args.refine_strength or gen_cfg.get("refine_strength", 0.35)),
        "guidance_scale": float(args.guidance_scale or gen_cfg.get("guidance_scale", 5.0)),
        "max_sequence_length": int(gen_cfg.get("max_sequence_length", 512)),
        "output_type": "np",
    }
    if args.hr_denoise_mode == "tiled":
        refined = refine_wan_tiled_from_latents(
            pipe,
            z_noisy,
            tile_hw=parse_hw(args.tile_latent_hw),
            overlap=int(args.tile_overlap),
            **refine_kwargs,
        )
    else:
        refined = refine_wan_from_latents(
            pipe,
            z_noisy,
            height=height,
            width=width,
            num_frames=num_frames,
            **refine_kwargs,
        )
    final_path = export_video(refined, out_dir / "lsrna_refined.mp4", fps=args.fps)
    print(f"Wrote refined Wan video to {final_path}")

    if args.save_intermediates:
        ref_frames = decode_wan_latents(pipe, z_ref, output_type="np")
        ref_path = export_video(ref_frames, out_dir / "lsr_reference.mp4", fps=args.fps)
        print(f"Wrote LSR reference video to {ref_path}")
        if z_hr_target is not None:
            target_frames = decode_wan_latents(pipe, z_hr_target, output_type="np")
            target_path = export_video(target_frames, out_dir / "target_reconstruction.mp4", fps=args.fps)
            print(f"Wrote target reconstruction video to {target_path}")


if __name__ == "__main__":
    main()
