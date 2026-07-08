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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--lr-latent", default="", help="Optional cached low-res latent tensor for sampler bring-up.")
    return parser.parse_args()


def load_lsr(checkpoint: str | Path, device: torch.device, dtype: torch.dtype):
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_video_lsr(payload["config"].get("model", {}))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device=device, dtype=dtype).eval()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("generation", {}).get("seed", 1234)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(config.get("model", {}).get("torch_dtype", "bf16"))
    out_dir = ensure_dir(config.get("output", {}).get("dir", "outputs/infer_lsrna"))

    lsr = load_lsr(config["lsr"]["checkpoint"], device, dtype)

    if args.lr_latent:
        item = torch.load(args.lr_latent, map_location="cpu")
        z_lr = item["latents"] if isinstance(item, dict) else item
        if z_lr.ndim == 4:
            z_lr = z_lr.unsqueeze(0)
        z_lr = z_lr.to(device=device, dtype=dtype)
    else:
        raise NotImplementedError(
            "Low-res Wan generation is intentionally left as a server integration step. "
            "Pass --lr-latent to test Video-LSR/RNA now, or wire the local WanPipeline to return latents."
        )

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

    torch.save({"z_lr": z_lr.cpu(), "z_hr_ref": z_ref.cpu(), "detail_map": detail.cpu(), "z_noisy": z_noisy.cpu()}, out_dir / "lsrna_latents.pt")
    print(f"Wrote Video-LSRNA latents to {out_dir / 'lsrna_latents.pt'}")
    print("Next server step: feed `z_noisy` into the Wan sampler at the configured refine timestep.")


if __name__ == "__main__":
    main()
