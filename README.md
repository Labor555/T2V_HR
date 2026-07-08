# T2V_HR

Lightweight Video-LSRNA experiments for testing whether a frozen Wan2.1-T2V-1.3B pipeline can obtain stable 4K-style generation from a low-resolution T2V reference.

The first implementation intentionally follows the original LSRNA spirit:

```text
low-res T2V reference latent
  -> Video-LSR
  -> Video-RNA
  -> frozen Wan high-resolution denoise
```

Only `Video-LSR` is trained. Wan, the VAE, and the text encoder stay frozen.

## Layout

- `src/t2v_hr/models/video_lsr.py`: small temporal latent upsampler.
- `src/t2v_hr/rna/video_rna.py`: heuristic region-time noise map and latent renoising.
- `src/t2v_hr/data/latent_dataset.py`: paired LR/HR latent cache dataset.
- `src/t2v_hr/wan/`: Wan VAE/video helpers.
- `scripts/cache_wan_latents.py`: build LR/HR latent pairs from videos.
- `scripts/train_lsr.py`: train the Video-LSR module.
- `scripts/infer_lsrna.py`: low-res Wan generation + Video-LSR/RNA + frozen high-res denoise scaffold.
- `remote/`: sagi sync/run helpers.

## Quick Local Smoke

The model and loss can be tested without Wan:

```bash
python -m pytest
python scripts/train_lsr.py --config configs/train_lsr_smoke.yaml --dry-run
```

## Server Workflow

Edit `remote/runtime.env.example`, save it as `remote/runtime.env`, then:

```bash
bash remote/sync_to_sagi.sh
bash remote/bootstrap_sagi.sh
bash remote/run_on_sagi.sh configs/train_lsr_wan13b.yaml
```

`sagi` must resolve in SSH config on the machine running the sync script.

## Current Wan Denoise Hook Status

`scripts/infer_lsrna.py` currently validates the Video-LSR/RNA half of the pipeline and writes:

```text
z_lr
z_hr_ref
detail_map
z_noisy
```

The last step, frozen Wan high-resolution denoising from `z_noisy`, needs a small server-side sampler hook because stock `WanPipeline(latents=...)` treats `latents` as initial noise, not as a partially denoised reference state. See `references/wan_sampler_hook.md`.
