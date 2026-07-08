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
bash remote/sync_to_sagi.sh
bash remote/bootstrap_sagi.sh
bash remote/run_remote.sh "python scripts/build_video_manifest.py --video-root /data/dataset/VideoSR/HQ-VSR --output /data/user/lbzhu/T2V_HR/data/sagittarius_video_manifest.jsonl --prompt 'high quality video'"
bash remote/run_remote.sh "python scripts/cache_wan_latents.py --config configs/cache_wan_latents_smoke.yaml --limit 2"
bash remote/run_on_sagi.sh configs/train_lsr_wan_smoke.yaml
bash remote/run_remote.sh "python scripts/build_video_rna.py --config configs/video_rna_smoke.yaml --latent /data/user/lbzhu/T2V_HR/cache/wan13b_latents_smoke/hr/00000000.pt"
bash remote/run_remote.sh "python scripts/infer_lsrna.py --config configs/infer_lsrna_smoke.yaml --lr-latent /data/user/lbzhu/T2V_HR/cache/wan13b_latents_smoke/lr/00000000.pt"
```

`infer_lsrna.py` currently writes `z_hr_ref`, `detail_map`, and `z_noisy`; the final Wan high-resolution sampler hook should be wired on sagi after checking the installed diffusers WanPipeline API.

After the smoke path passes, switch `cache_wan_latents.yaml` and `train_lsr_wan13b.yaml` to the full 720p-to-4K latent cache. Keep Video-RNA standalone as a cheap diagnostic before spending time on frozen HR denoise.
