# Wan2.1-1.3B Video-LSRNA Experiment Plan

## Goal

Test whether an LSRNA-faithful video variant can provide a stable high-resolution reference for frozen Wan high-resolution denoising.

## Minimal Gates

1. `Video-LSR` beats latent bicubic on cached LR/HR Wan VAE latent pairs.
2. Temporal difference loss reduces flicker relative to frame-wise upsampling.
3. Video-RNA improves local detail after frozen HR denoise without changing low-frequency structure.
4. The Wan sampler can continue from the constructed high-resolution latent state.

## First Server Run

```bash
python scripts/build_video_manifest.py --video-root /data/labor/datasets/videos --output /data/labor/datasets/sagi_video_manifest.jsonl
python scripts/cache_wan_latents.py --config configs/cache_wan_latents.yaml --limit 128
python scripts/train_lsr.py --config configs/train_lsr_wan13b.yaml
python scripts/infer_lsrna.py --config configs/infer_lsrna_wan13b.yaml --lr-latent /data/labor/T2V_HR/cache/wan13b_latents/lr/00000000.pt
```

`infer_lsrna.py` currently writes `z_hr_ref`, `detail_map`, and `z_noisy`; the final Wan high-resolution sampler hook should be wired on sagi after checking the installed diffusers WanPipeline API.
