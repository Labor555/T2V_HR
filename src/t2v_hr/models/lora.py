from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        update = F.linear(F.linear(self.dropout(x).float(), self.lora_A), self.lora_B)
        return result + update.to(dtype=result.dtype) * self.scaling


@dataclass
class LoRAInjectionResult:
    module_names: list[str]
    trainable_parameters: int


def _matches(name: str, target_substrings: tuple[str, ...]) -> bool:
    return any(target in name for target in target_substrings)


def inject_lora_linear(
    module: nn.Module,
    *,
    target_substrings: list[str] | tuple[str, ...],
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> LoRAInjectionResult:
    targets = tuple(str(target) for target in target_substrings)
    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _matches(full_name, targets):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(module)
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return LoRAInjectionResult(module_names=replaced, trainable_parameters=trainable)


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad and ("lora_A" in name or "lora_B" in name)
    }


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor], *, strict: bool = True) -> None:
    own = dict(module.named_parameters())
    missing: list[str] = []
    for name, tensor in state.items():
        if name not in own:
            if strict:
                raise KeyError(f"LoRA parameter not found in target module: {name}")
            continue
        own[name].data.copy_(tensor.to(device=own[name].device, dtype=own[name].dtype))
    if strict:
        for name, parameter in own.items():
            if parameter.requires_grad and ("lora_A" in name or "lora_B" in name) and name not in state:
                missing.append(name)
        if missing:
            raise KeyError(f"Missing LoRA parameters: {missing[:8]} (total {len(missing)})")
