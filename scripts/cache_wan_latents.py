#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
from t2v_hr.wan.vae import encode_video_latents, load_wan_vae


def _latent_file_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        item = torch.load(path, map_location="cpu")
        tensor = item["latents"] if isinstance(item, dict) else item
        return isinstance(tensor, torch.Tensor) and tensor.numel() > 0
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return False


def _atomic_torch_save(obj, path: Path) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--index-name", default="index.jsonl")
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
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")
    indexed_rows = [(idx, row) for idx, row in enumerate(rows) if idx % args.num_shards == args.shard_id]

    vae = load_wan_vae(
        config["model"]["model_path"],
        fmt=str(config["model"].get("format", "diffusers")).lower(),
        vae_weight=str(config["model"].get("vae_weight", "Wan2.1_VAE.pth")),
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

    index_path = out_dir / args.index_name
    index_rows = []
    for idx, row in tqdm(indexed_rows, desc=f"cache_wan_latents[{args.shard_id}/{args.num_shards}]"):
        source = str(row[video_key])
        stem = f"{idx:08d}"
        lr_path = lr_dir / f"{stem}.pt"
        hr_path = hr_dir / f"{stem}.pt"
        if not _latent_file_ok(lr_path) or not _latent_file_ok(hr_path):
            hr_video = load_video_tensor(source, num_frames=num_frames, size=hr_size, device=device, dtype=dtype)
            lr_video = load_video_tensor(source, num_frames=num_frames, size=lr_size, device=device, dtype=dtype)
            hr_latents = encode_video_latents(vae, hr_video).cpu()
            lr_latents = encode_video_latents(vae, lr_video).cpu()
            _atomic_torch_save(
                {"latents": hr_latents.squeeze(0), "source": source, "prompt": row.get(prompt_key, "")},
                hr_path,
            )
            _atomic_torch_save(
                {"latents": lr_latents.squeeze(0), "source": source, "prompt": row.get(prompt_key, "")},
                lr_path,
            )
            del hr_video, lr_video, hr_latents, lr_latents
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        index_rows.append(
            {
                "source": source,
                "prompt": row.get(prompt_key, ""),
                "lr_latent": str(lr_path),
                "hr_latent": str(hr_path),
            }
        )
        write_jsonl(index_rows, index_path)

    write_jsonl(index_rows, index_path)
    print(f"Wrote {len(index_rows)} latent pairs to {index_path}")


if __name__ == "__main__":
    main()
