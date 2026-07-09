#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from t2v_hr.data.latent_dataset import LatentPairDataset
from t2v_hr.models.lora import inject_lora_linear, lora_state_dict, load_lora_state_dict
from t2v_hr.models.losses import (
    spatial_blur,
    spatial_gradients,
    spatial_high_frequency,
    spatial_laplacian,
    temporal_difference,
)
from t2v_hr.models.video_lsr import build_video_lsr
from t2v_hr.utils.config import ensure_dir, load_config
from t2v_hr.utils.torch_utils import autocast_context, dtype_from_string, seed_everything
from t2v_hr.wan.pipeline import load_wan_pipeline_legacy, load_wan_text_encoder_legacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", default="")
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


def _load_lsrna(config: dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.nn.Module | None:
    lsr_cfg = config.get("lsrna", {})
    if not bool(lsr_cfg.get("enabled", True)):
        return None
    checkpoint_path = Path(lsr_cfg["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = lsr_cfg.get("model") or payload.get("config", {}).get("model", {})
    model = build_video_lsr(model_cfg)
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def _prepare_prompt_embeds_cpu(
    model_path: str | Path,
    prompt: str,
    *,
    max_sequence_length: int,
    dtype: torch.dtype,
    target_device: torch.device,
) -> torch.Tensor:
    import gc

    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(Path(model_path) / "google" / "umt5-xxl"), local_files_only=True)
    text_encoder = load_wan_text_encoder_legacy(model_path, torch_dtype=torch.float32, device="cpu")
    cleaned = [prompt_clean(prompt)]
    text_inputs = tokenizer(
        cleaned,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    mask = text_inputs.attention_mask
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state.to(dtype=dtype)
    prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
        dim=0,
    )
    del text_encoder, tokenizer
    gc.collect()
    return prompt_embeds.to(device=target_device, dtype=dtype)


def _broadcast_sigma(sigmas: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return sigmas.view(-1, 1, 1, 1, 1).to(device=like.device, dtype=like.dtype)


def _flow_noisy_sample(clean: torch.Tensor, sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(clean)
    sigma = _broadcast_sigma(sigmas, clean)
    noisy = (1.0 - sigma) * clean + sigma * noise
    target_velocity = noise - clean
    return noisy, noise, target_velocity


def _loss_dict(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    coarse_x0: torch.Tensor | None,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    enabled = {key for key, value in weights.items() if float(value) != 0.0}
    if "velocity_l2" in enabled:
        losses["velocity_l2"] = F.mse_loss(pred_velocity, target_velocity)
    if "velocity_l1" in enabled:
        losses["velocity_l1"] = F.l1_loss(pred_velocity, target_velocity)
    if "x0_l1" in enabled:
        losses["x0_l1"] = F.l1_loss(pred_x0, target_x0)
    if "temporal_l1" in enabled:
        losses["temporal_l1"] = F.l1_loss(temporal_difference(pred_x0), temporal_difference(target_x0))
    if "highfreq_l1" in enabled:
        losses["highfreq_l1"] = F.l1_loss(spatial_high_frequency(pred_x0), spatial_high_frequency(target_x0))
    if "spatial_grad_l1" in enabled:
        pred_gh, pred_gw = spatial_gradients(pred_x0)
        target_gh, target_gw = spatial_gradients(target_x0)
        losses["spatial_grad_l1"] = 0.5 * (F.l1_loss(pred_gh, target_gh) + F.l1_loss(pred_gw, target_gw))
    if "laplacian_l1" in enabled:
        losses["laplacian_l1"] = F.l1_loss(spatial_laplacian(pred_x0), spatial_laplacian(target_x0))
    if "coarse_lowfreq_l1" in enabled and coarse_x0 is not None:
        losses["coarse_lowfreq_l1"] = F.l1_loss(spatial_blur(pred_x0), spatial_blur(coarse_x0))
    total = pred_x0.new_tensor(0.0)
    for key, value in losses.items():
        total = total + float(weights.get(key, 0.0)) * value
    losses["total"] = total
    return losses


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    transformer: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    injected_modules: list[str],
) -> None:
    module = transformer.module if isinstance(transformer, DistributedDataParallel) else transformer
    payload = {
        "step": step,
        "lora": lora_state_dict(module),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "injected_modules": injected_modules,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 2026)))
    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_string(config.get("precision", "bf16"))

    out_dir = ensure_dir(args.output_dir or config.get("output_dir", "outputs/hr_renoise_lora"))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")

    data_cfg = config.get("data", {})
    crop_cfg = data_cfg.get("random_crop", {})
    hr_size = tuple(crop_cfg.get("hr_size", [96, 160]))
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

    model_cfg = config.get("model", {})
    pipe = load_wan_pipeline_legacy(
        model_cfg["model_path"],
        torch_dtype=dtype,
        device=device,
        vae_format=str(model_cfg.get("format", "legacy")),
        vae_weight=str(model_cfg.get("vae_weight", "Wan2.1_VAE.pth")),
        flow_shift=float(model_cfg.get("flow_shift", 5.0)),
        load_text_encoder=False,
        load_transformer=True,
        load_vae=False,
    )
    prompt_cfg = config.get("prompt", {})
    prompt = str(prompt_cfg.get("prompt_override", "high"))
    prompt_embeds_1 = _prepare_prompt_embeds_cpu(
        model_cfg["model_path"],
        prompt,
        max_sequence_length=int(prompt_cfg.get("max_sequence_length", 512)),
        dtype=dtype,
        target_device=device,
    )
    torch.cuda.empty_cache()

    transformer = pipe.transformer
    transformer.train()
    if bool(config.get("train", {}).get("gradient_checkpointing", True)) and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    lora_cfg = config.get("lora", {})
    result = inject_lora_linear(
        transformer,
        target_substrings=tuple(lora_cfg.get("target_substrings", ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"])),
        rank=int(lora_cfg.get("rank", 8)),
        alpha=float(lora_cfg.get("alpha", 8.0)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    if is_main:
        print(f"injected_lora_modules={len(result.module_names)} trainable_parameters={result.trainable_parameters}")
        print("injected_lora_head=" + ",".join(result.module_names[:12]))

    lsrna = _load_lsrna(config, device, dtype)
    trainable = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("train", {}).get("lr", 1e-4)),
        weight_decay=float(config.get("train", {}).get("weight_decay", 0.0)),
    )

    start_step = 0
    latest_ckpt = ckpt_dir / "latest.pt"
    if bool(config.get("train", {}).get("resume", True)) and not args.no_resume and latest_ckpt.exists():
        payload = torch.load(latest_ckpt, map_location="cpu")
        load_lora_state_dict(transformer, payload["lora"], strict=True)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload.get("step", 0))
        if is_main:
            print(f"Resumed {latest_ckpt} at step {start_step}")

    if distributed:
        transformer = DistributedDataParallel(transformer, device_ids=[local_rank], output_device=local_rank)

    train_cfg = config.get("train", {})
    loss_weights = {key: float(value) for key, value in config.get("loss", {}).items()}
    max_steps = int(args.max_steps or train_cfg.get("max_steps", 50000))
    log_every = int(train_cfg.get("log_every", 20))
    save_every = int(train_cfg.get("save_every", 500))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    num_train_timesteps = int(model_cfg.get("num_train_timesteps", 1000))
    sigma_min = float(train_cfg.get("sigma_min", 0.12))
    sigma_max = float(train_cfg.get("sigma_max", 0.55))

    step = start_step
    if step >= max_steps:
        if is_main:
            print(f"Already reached max_steps={max_steps}; nothing to do.")
        cleanup_distributed(distributed)
        return

    progress = tqdm(total=max_steps, initial=step, desc="train_hr_renoise_lora", disable=not is_main)
    epoch = 0
    accum_index = 0
    optimizer.zero_grad(set_to_none=True)
    while step < max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            lr = batch["lr"].to(device=device, dtype=dtype)
            hr = batch["hr"].to(device=device, dtype=dtype)
            with torch.no_grad():
                coarse = lsrna(lr) if lsrna is not None else None
            batch_size = hr.shape[0]
            sigmas = torch.empty(batch_size, device=device, dtype=torch.float32).uniform_(sigma_min, sigma_max)
            noisy, _noise, target_velocity = _flow_noisy_sample(hr.float(), sigmas)
            prompt_embeds = prompt_embeds_1.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
            timesteps = (sigmas * float(num_train_timesteps)).to(device=device, dtype=torch.float32)

            with autocast_context(device, dtype):
                pred_velocity = transformer(
                    hidden_states=noisy.to(dtype=dtype),
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                pred_velocity_f = pred_velocity.float()
                pred_x0 = noisy - _broadcast_sigma(sigmas, noisy) * pred_velocity_f
                losses = _loss_dict(
                    pred_velocity_f,
                    target_velocity.float(),
                    pred_x0.float(),
                    hr.float(),
                    coarse.float() if coarse is not None else None,
                    loss_weights,
                )
                loss = losses["total"] / grad_accum
            loss.backward()
            accum_index += 1
            if accum_index % grad_accum != 0:
                continue
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step += 1
            if is_main:
                progress.update(1)
            if is_main and (step % log_every == 0 or step == 1):
                progress.set_postfix({key: f"{value.item():.4f}" for key, value in losses.items()})
            if is_main and (step % save_every == 0 or step == max_steps):
                _save_checkpoint(
                    ckpt_dir / f"step_{step:08d}.pt",
                    step=step,
                    transformer=transformer,
                    optimizer=optimizer,
                    config=config,
                    injected_modules=result.module_names,
                )
                _save_checkpoint(
                    ckpt_dir / "latest.pt",
                    step=step,
                    transformer=transformer,
                    optimizer=optimizer,
                    config=config,
                    injected_modules=result.module_names,
                )
            if step >= max_steps:
                break
        epoch += 1
    if is_main:
        progress.close()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
