#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from tqdm import tqdm

from t2v_hr.data.manifest import read_manifest, write_jsonl
from t2v_hr.data.video_io import load_video_tensor
from t2v_hr.utils.config import ensure_dir, load_config
from t2v_hr.utils.torch_utils import dtype_from_string
from t2v_hr.wan.vae import encode_video_latents, load_wan_vae_from_diffusers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    runtime_cfg = config.get("runtime", {})
    device = torch.device(runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = dtype_from_string(config.get("model", {}).get("torch_dtype", "bf16"))
    out_dir = ensure_dir(data_cfg["output_dir"])
    lr_dir = ensure_dir(out_dir / "lr")
    hr_dir = ensure_dir(out_dir / "hr")

    rows = read_manifest(data_cfg["manifest"])
    limit = args.limit if args.limit is not None else int(data_cfg.get("limit", 0))
    if limit and limit > 0:
        rows = rows[:limit]

    vae = load_wan_vae_from_diffusers(
        config["model"]["model_path"],
        torch_dtype=dtype,
        device=device,
        enable_tiling=bool(runtime_cfg.get("vae_tiling", True)),
        enable_slicing=bool(runtime_cfg.get("vae_slicing", True)),
    )

    video_key = data_cfg.get("video_key", "video")
    prompt_key = data_cfg.get("prompt_key", "prompt")
    num_frames = int(data_cfg.get("num_frames", 81))
    hr_size = tuple(int(v) for v in data_cfg.get("hr_size", [2160, 3840]))
    lr_size = tuple(int(v) for v in data_cfg.get("lr_size", [720, 1280]))

    index_rows = []
    for idx, row in enumerate(tqdm(rows, desc="cache_wan_latents")):
        source = str(row[video_key])
        stem = f"{idx:08d}"
        lr_path = lr_dir / f"{stem}.pt"
        hr_path = hr_dir / f"{stem}.pt"
        if not lr_path.exists() or not hr_path.exists():
            hr_video = load_video_tensor(source, num_frames=num_frames, size=hr_size, device=device, dtype=dtype)
            lr_video = load_video_tensor(source, num_frames=num_frames, size=lr_size, device=device, dtype=dtype)
            hr_latents = encode_video_latents(vae, hr_video).cpu()
            lr_latents = encode_video_latents(vae, lr_video).cpu()
            torch.save({"latents": hr_latents.squeeze(0), "source": source, "prompt": row.get(prompt_key, "")}, hr_path)
            torch.save({"latents": lr_latents.squeeze(0), "source": source, "prompt": row.get(prompt_key, "")}, lr_path)
        index_rows.append(
            {
                "source": source,
                "prompt": row.get(prompt_key, ""),
                "lr_latent": str(lr_path),
                "hr_latent": str(hr_path),
            }
        )

    write_jsonl(index_rows, out_dir / "index.jsonl")
    print(f"Wrote {len(index_rows)} latent pairs to {out_dir}")


if __name__ == "__main__":
    main()

