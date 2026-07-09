#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from t2v_hr.data.latent_dataset import LatentPairDataset
from t2v_hr.models.losses import video_lsr_loss
from t2v_hr.models.video_lsr import build_video_lsr
from t2v_hr.utils.config import ensure_dir, load_config
from t2v_hr.utils.torch_utils import autocast_context, dtype_from_string, seed_everything


class SyntheticLatentDataset(Dataset):
    def __init__(
        self,
        length: int = 8,
        channels: int = 16,
        frames: int = 5,
        lr_hw: tuple[int, int] = (16, 16),
        scale_factor: int = 2,
    ) -> None:
        self.length = length
        self.channels = channels
        self.frames = frames
        self.lr_hw = lr_hw
        self.scale_factor = int(scale_factor)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index)
        lr = torch.randn(self.channels, self.frames, *self.lr_hw, generator=generator)
        hr = torch.nn.functional.interpolate(
            lr.permute(1, 0, 2, 3), scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        ).permute(1, 0, 2, 3)
        hr = hr + 0.03 * torch.randn(hr.shape, generator=generator)
        return {"lr": lr, "hr": hr, "prompt": "", "source": f"synthetic-{index}"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic latent pairs and run only a few steps.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoints/latest.pt.")
    return parser.parse_args()


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    return world_size > 1, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)))
    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(config.get("precision", "fp32"))
    out_dir = ensure_dir(config.get("output_dir", "outputs/lsr"))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    model = build_video_lsr(config.get("model", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("train", {}).get("lr", 2e-4)))
    start_step = 0
    resume_enabled = bool(config.get("train", {}).get("resume", True)) and not args.no_resume
    latest_ckpt = ckpt_dir / "latest.pt"
    if resume_enabled and latest_ckpt.exists():
        payload = torch.load(latest_ckpt, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        if is_main:
            print(f"Resumed {latest_ckpt} at step {start_step}")
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    data_cfg = config.get("data", {})
    if args.dry_run or not data_cfg.get("cache_index"):
        model_cfg = config.get("model", {})
        dataset: Dataset = SyntheticLatentDataset(
            channels=int(model_cfg.get("in_channels", 16)),
            scale_factor=int(model_cfg.get("scale_factor", 2)),
        )
    else:
        crop_cfg = data_cfg.get("random_crop", {})
        hr_size = tuple(crop_cfg.get("hr_size", [64, 64]))
        dataset = LatentPairDataset(
            data_cfg["cache_index"],
            random_crop=bool(crop_cfg.get("enabled", True)),
            hr_crop_size=(int(hr_size[0]), int(hr_size[1])),
        )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
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
    step = start_step
    if step >= max_steps:
        if is_main:
            print(f"Already reached max_steps={max_steps}; nothing to do.")
        cleanup_distributed(distributed)
        return
    progress = tqdm(total=max_steps, initial=step, desc="train_lsr", disable=not is_main)
    epoch = 0
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
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
            if is_main:
                progress.update(1)
            if is_main and (step % log_every == 0 or step == 1):
                progress.set_postfix({key: f"{value.item():.4f}" for key, value in losses.items()})
            if is_main and (step % save_every == 0 or step == max_steps):
                model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
                payload = {
                    "model": model_to_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                }
                torch.save(payload, ckpt_dir / f"step_{step:08d}.pt")
                torch.save(payload, ckpt_dir / "latest.pt")
            if step >= max_steps:
                break
        epoch += 1
    if is_main:
        progress.close()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
