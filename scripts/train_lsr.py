#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from t2v_hr.data.latent_dataset import LatentPairDataset
from t2v_hr.models.losses import video_lsr_loss
from t2v_hr.models.video_lsr import build_video_lsr
from t2v_hr.utils.config import ensure_dir, load_config
from t2v_hr.utils.torch_utils import autocast_context, dtype_from_string, seed_everything


class SyntheticLatentDataset(Dataset):
    def __init__(self, length: int = 8, channels: int = 16, frames: int = 5, lr_hw: tuple[int, int] = (16, 16)) -> None:
        self.length = length
        self.channels = channels
        self.frames = frames
        self.lr_hw = lr_hw

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index)
        lr = torch.randn(self.channels, self.frames, *self.lr_hw, generator=generator)
        hr = torch.nn.functional.interpolate(
            lr.permute(1, 0, 2, 3), scale_factor=2, mode="bilinear", align_corners=False
        ).permute(1, 0, 2, 3)
        hr = hr + 0.03 * torch.randn(hr.shape, generator=generator)
        return {"lr": lr, "hr": hr, "prompt": "", "source": f"synthetic-{index}"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic latent pairs and run only a few steps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(config.get("precision", "fp32"))
    out_dir = ensure_dir(config.get("output_dir", "outputs/lsr"))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    model = build_video_lsr(config.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("train", {}).get("lr", 2e-4)))

    data_cfg = config.get("data", {})
    if args.dry_run or not data_cfg.get("cache_index"):
        dataset: Dataset = SyntheticLatentDataset(channels=int(config.get("model", {}).get("in_channels", 16)))
    else:
        crop_cfg = data_cfg.get("random_crop", {})
        hr_size = tuple(crop_cfg.get("hr_size", [64, 64]))
        dataset = LatentPairDataset(
            data_cfg["cache_index"],
            random_crop=bool(crop_cfg.get("enabled", True)),
            hr_crop_size=(int(hr_size[0]), int(hr_size[1])),
        )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=True,
    )

    max_steps = int(config.get("train", {}).get("max_steps", 1000))
    if args.dry_run:
        max_steps = min(max_steps, 3)
    log_every = int(config.get("train", {}).get("log_every", 50))
    save_every = int(config.get("train", {}).get("save_every", 5000))
    grad_clip = float(config.get("train", {}).get("grad_clip", 0.0))
    loss_weights = config.get("loss", {})

    model.train()
    step = 0
    progress = tqdm(total=max_steps, desc="train_lsr")
    while step < max_steps:
        for batch in loader:
            lr = batch["lr"].to(device=device, dtype=dtype)
            hr = batch["hr"].to(device=device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                pred = model(lr)
                losses = video_lsr_loss(pred.float(), hr.float(), lr.float(), loss_weights)
            losses["total"].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            progress.update(1)
            if step % log_every == 0 or step == 1:
                progress.set_postfix({key: f"{value.item():.4f}" for key, value in losses.items()})
            if step % save_every == 0 or step == max_steps:
                payload = {"model": model.state_dict(), "config": config, "step": step}
                torch.save(payload, ckpt_dir / f"step_{step:08d}.pt")
                torch.save(payload, ckpt_dir / "latest.pt")
            if step >= max_steps:
                break
    progress.close()


if __name__ == "__main__":
    main()

