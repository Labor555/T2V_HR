# Wan Sampler Hook Notes

`diffusers.WanPipeline` accepts a `latents` argument, but in the stock pipeline that tensor is treated as the initial noise at the beginning of the schedule. Video-LSRNA needs a slightly different hook:

```text
z_hr_ref --Video-RNA--> z_t_ref
run Wan denoising from timestep t_ref to 0
```

The server-side integration should therefore do one of these:

1. Add a `start_timestep` / `denoising_start` argument to the local Wan pipeline.
2. Copy the pipeline denoising loop and replace the initial `latents` plus `timesteps` slice.
3. Use the installed diffusers callback only to inspect/capture latents, not as the final refinement mechanism.

Expected first implementation:

```python
timesteps = scheduler.timesteps[start_index:]
latents = z_noisy
for t in timesteps:
    noise_pred = transformer(latents, timestep=t, encoder_hidden_states=prompt_embeds, ...)
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

Keep this hook local to `scripts/infer_lsrna.py` or a small `src/t2v_hr/wan/sampler.py` helper after checking the exact installed diffusers version on sagi.

