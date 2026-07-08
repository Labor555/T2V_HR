#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from t2v_hr.rna.video_rna import VideoRNAConfig, apply_video_rna
from t2v_hr.utils.config import ensure_dir, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--latent", required=True, help="Tensor or dict containing z_hr_ref/latents/z.")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _load_latents(path: str | Path) -> torch.Tensor:
    item = torch.load(path, map_location="cpu")
    if isinstance(item, torch.Tensor):
        tensor = item
    elif isinstance(item, dict):
        for key in ("z_hr_ref", "latents", "latent", "z"):
            if key in item:
                tensor = item[key]
                break
        else:
            raise KeyError(f"No latent tensor key found in {path}")
    else:
        raise TypeError(f"Unsupported latent object: {type(item)!r}")
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"Expected latent [B,C,T,H,W] or [C,T,H,W], got {tuple(tensor.shape)}")
    return tensor


def _save_preview(detail: torch.Tensor, output_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    # Contact sheet over up to 8 frames.
    detail = detail[0, 0].detach().float().cpu().clamp(0, 1)
    frames = []
    count = min(8, detail.shape[0])
    for idx in torch.linspace(0, detail.shape[0] - 1, count).long().tolist():
        arr = (detail[idx].numpy() * 255).astype("uint8")
        frames.append(Image.fromarray(arr, mode="L").convert("RGB"))
    width = sum(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    sheet = Image.new("RGB", (width, height), "black")
    x = 0
    for frame in frames:
        sheet.paste(frame, (x, 0))
        x += frame.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_dir = ensure_dir(args.output_dir or config.get("output", {}).get("dir", "outputs/video_rna"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latents = _load_latents(args.latent).to(device=device)

    rna_cfg = config.get("rna", {})
    cfg = VideoRNAConfig(
        latent_weight=float(rna_cfg.get("latent_weight", 0.4)),
        temporal_weight=float(rna_cfg.get("temporal_weight", 0.3)),
        edge_weight=float(rna_cfg.get("edge_weight", 0.0)),
        smooth_kernel=int(rna_cfg.get("smooth_kernel", 3)),
        gamma=float(rna_cfg.get("gamma", 1.0)),
        base_strength=float(rna_cfg.get("base_strength", 0.02)),
        detail_strength=float(rna_cfg.get("detail_strength", 0.18)),
    )
    noised, detail = apply_video_rna(latents, config=cfg)
    torch.save(
        {
            "z_hr_ref": latents.cpu(),
            "detail_map": detail.cpu(),
            "z_noisy": noised.cpu(),
            "rna_config": cfg.__dict__,
        },
        out_dir / "video_rna.pt",
    )
    _save_preview(detail, out_dir / "video_rna_preview.png")
    print(f"Wrote Video-RNA outputs to {out_dir}")


if __name__ == "__main__":
    main()

