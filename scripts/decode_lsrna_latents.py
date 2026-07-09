#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from t2v_hr.utils.config import load_config
from t2v_hr.utils.torch_utils import dtype_from_string
from t2v_hr.wan.pipeline import decode_wan_latents, export_video, load_wan_pipeline_legacy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--latents", required=True, help="Path to lsrna_latents.pt.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--keys",
        default="z_hr_ref,z_noisy,z_hr_target",
        help="Comma-separated latent payload keys to decode if present.",
    )
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def video_name_for_key(key: str) -> str:
    names = {
        "z_hr_ref": "lsr_reference.mp4",
        "z_noisy": "lsrna_noisy.mp4",
        "z_hr_target": "target_reconstruction.mp4",
        "z_lr": "lowres_reconstruction.mp4",
    }
    return names.get(key, f"{key}.mp4")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dtype = dtype_from_string(config.get("model", {}).get("torch_dtype", "bf16"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_wan_pipeline_legacy(
        config["model"]["model_path"],
        torch_dtype=dtype,
        device=device,
        vae_format=str(config["model"].get("format", "legacy")).lower(),
        vae_weight=str(config["model"].get("vae_weight", "Wan2.1_VAE.pth")),
        flow_shift=float(config.get("scheduler", {}).get("flow_shift", 5.0)),
        load_text_encoder=False,
        load_transformer=False,
        load_vae=True,
    )
    payload = torch.load(args.latents, map_location="cpu")
    for key in [item.strip() for item in args.keys.split(",") if item.strip()]:
        if key not in payload:
            print(f"skip_missing_key={key}", flush=True)
            continue
        latents = payload[key]
        if latents.ndim == 4:
            latents = latents.unsqueeze(0)
        print(f"decode_key={key} shape={tuple(latents.shape)}", flush=True)
        frames = decode_wan_latents(pipe, latents, output_type="np")
        path = export_video(frames, out_dir / video_name_for_key(key), fps=args.fps)
        print(f"wrote={path}", flush=True)
        del latents, frames
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
