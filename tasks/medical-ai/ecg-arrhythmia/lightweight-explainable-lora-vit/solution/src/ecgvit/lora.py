"""Low-Rank Adaptation (manuscript section 3.4, equations 14-15).

    y = Wx + b            (14)
    y = Wx + BAx + b      (15)   with A in R^{r x d_in}, B in R^{d_out x r}, r << min(d)

Implementation notes
--------------------
* `B` is zero-initialised, so at step 0 the adapted model is *functionally identical* to
  the frozen backbone. `test_model_lora.py::test_lora_init_is_identity` asserts this;
  a non-zero init would make the "frozen backbone" claim untestable.
* Freezing sets `requires_grad = False` on the base weight. It does NOT detach the input,
  so gradients still flow *through* a frozen layer to earlier LoRA modules -- which is
  required for a deep adapter stack to train at all. The test distinguishes these two
  things explicitly, because conflating them is the usual way a "LoRA" implementation ends
  up either training nothing or training everything.
* The scaling is `alpha / r`, applied to the low-rank product.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from .config import LoRAConfig


class LoRALinear(nn.Module):
    """Wraps an `nn.Linear`, freezing it and adding a trainable rank-r update."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # delta-W = 0 at init -> identity to the backbone

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def delta_weight(self) -> torch.Tensor:
        """B @ A * scaling -- the effective weight update, for inspection/merging."""
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> nn.Linear:
        """Fold the adapter into a plain Linear (deployment / inference-time export)."""
        merged = nn.Linear(self.in_features, self.out_features, bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        lora = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + lora * self.scaling

    def extra_repr(self) -> str:
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.3f}"


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------
def _iter_named_linears(module: nn.Module) -> Iterable[Tuple[nn.Module, str, nn.Linear]]:
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                yield parent, name, child


def inject_lora(model: nn.Module, cfg: LoRAConfig) -> List[str]:
    """Replace targeted `nn.Linear` layers with `LoRALinear`. Returns the names replaced."""
    if not cfg.enabled:
        return []
    targets = set(cfg.target_modules)
    replaced: List[str] = []
    # Snapshot first: mutating children while iterating modules() is undefined.
    candidates = [(p, n, c) for p, n, c in _iter_named_linears(model) if n in targets]
    for parent, name, child in candidates:
        setattr(
            parent,
            name,
            LoRALinear(
                child,
                rank=cfg.rank,
                alpha=cfg.alpha,
                dropout=cfg.dropout,
                freeze_base=cfg.freeze_backbone,
            ),
        )
        replaced.append(name)
    if not replaced:
        raise RuntimeError(
            f"LoRA target modules {sorted(targets)} matched no nn.Linear in the model. "
            "Check ModelConfig/LoRAConfig.target_modules against the module names."
        )
    return replaced


def freeze_backbone(model: nn.Module, cfg: LoRAConfig) -> None:
    """Freeze everything except LoRA matrices (and optionally the head / norms)."""
    for name, param in model.named_parameters():
        is_lora = ".lora_A" in name or ".lora_B" in name or name.endswith(("lora_A", "lora_B"))
        is_head = name.startswith("head.")
        is_norm = ".norm" in name or name.startswith("norm.")
        if is_lora:
            param.requires_grad_(True)
        elif is_head and cfg.train_head:
            param.requires_grad_(True)
        elif is_norm and cfg.train_norms:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


@dataclass
class ParameterAccounting:
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    lora_parameters: int
    baseline_trainable_parameters: int
    trainable_reduction_pct: float
    reduction_factor: float
    lora_rank: int
    lora_alpha: int
    lora_target_modules: List[str]
    n_lora_layers: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "lora_parameters": self.lora_parameters,
            "baseline_trainable_parameters": self.baseline_trainable_parameters,
            "trainable_reduction_pct": round(self.trainable_reduction_pct, 4),
            "reduction_factor": round(self.reduction_factor, 4),
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_target_modules": self.lora_target_modules,
            "n_lora_layers": self.n_lora_layers,
        }


def count_parameters(model: nn.Module, cfg: LoRAConfig) -> ParameterAccounting:
    """Parameter accounting for Table 8.

    `baseline_trainable_parameters` is the count for full fine-tuning of the SAME
    architecture without adapters -- i.e. every non-LoRA parameter. The reduction is
    reported against that, which is the only comparison that means anything.
    """
    total = 0
    trainable = 0
    lora_params = 0
    base_params = 0
    n_lora_layers = 0

    for module in model.modules():
        if isinstance(module, LoRALinear):
            n_lora_layers += 1

    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if ".lora_A" in name or ".lora_B" in name:
            lora_params += n
        else:
            base_params += n

    reduction = 100.0 * (1.0 - trainable / base_params) if base_params else 0.0
    factor = (base_params / trainable) if trainable else float("inf")
    return ParameterAccounting(
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
        lora_parameters=lora_params,
        baseline_trainable_parameters=base_params,
        trainable_reduction_pct=reduction,
        reduction_factor=factor,
        lora_rank=cfg.rank,
        lora_alpha=cfg.alpha,
        lora_target_modules=list(cfg.target_modules),
        n_lora_layers=n_lora_layers,
    )


__all__ = [
    "LoRALinear", "inject_lora", "freeze_backbone",
    "ParameterAccounting", "count_parameters",
]
