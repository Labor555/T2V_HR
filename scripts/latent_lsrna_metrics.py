#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from t2v_hr.models.losses import spatial_gradients, spatial_high_frequency, temporal_difference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", required=True, help="Path to lsrna_latents.pt.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    return parser.parse_args()


def l1(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.l1_loss(a.float(), b.float()).item())


def main() -> None:
    args = parse_args()
    payload = torch.load(args.latents, map_location="cpu")
    target = payload["z_hr_target"].float()
    lsrna = payload["z_noisy"].float()
    ref = payload.get("z_hr_ref")
    metrics: dict[str, float | str] = {
        "latent_l1_lsrna_to_wan": l1(lsrna, target),
        "temporal_l1_lsrna_to_wan": l1(temporal_difference(lsrna), temporal_difference(target)),
        "highfreq_l1_lsrna_to_wan": l1(spatial_high_frequency(lsrna), spatial_high_frequency(target)),
    }
    pred_gh, pred_gw = spatial_gradients(lsrna)
    target_gh, target_gw = spatial_gradients(target)
    metrics["spatial_grad_l1_lsrna_to_wan"] = 0.5 * (l1(pred_gh, target_gh) + l1(pred_gw, target_gw))
    if ref is not None:
        ref = ref.float()
        metrics["latent_l1_lsr_ref_to_wan"] = l1(ref, target)
        metrics["highfreq_l1_lsr_ref_to_wan"] = l1(spatial_high_frequency(ref), spatial_high_frequency(target))
        ref_gh, ref_gw = spatial_gradients(ref)
        metrics["spatial_grad_l1_lsr_ref_to_wan"] = 0.5 * (l1(ref_gh, target_gh) + l1(ref_gw, target_gw))
        metrics["delta_highfreq_lsrna_minus_lsr_ref"] = (
            metrics["highfreq_l1_lsrna_to_wan"] - metrics["highfreq_l1_lsr_ref_to_wan"]
        )
        metrics["delta_spatial_grad_lsrna_minus_lsr_ref"] = (
            metrics["spatial_grad_l1_lsrna_to_wan"] - metrics["spatial_grad_l1_lsr_ref_to_wan"]
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
