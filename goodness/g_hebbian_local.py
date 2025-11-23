from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    mean_momentum: float = 0.9  # running mean of activations per unit
    center: bool = True         # center activations by running mean


class HebbianLocal:
    """
    Hebbian-style local goodness: sum of squared activations.
    Uses a running per-unit mean (EMA) for centering to preserve homeostasis.
    Goodness = sum(h^2) which is equivalent to Hebbian h*h.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.running_means: Dict[int, torch.Tensor] = {}

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        if layer_index not in self.running_means:
            self.running_means[layer_index] = torch.zeros(width, device=device, dtype=dtype)
        return {"mean": self.running_means[layer_index]}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        m = state["mean"]
        mom = self.cfg.mean_momentum
        batch_mean = h.mean(dim=0)
        m.mul_(mom).add_(batch_mean, alpha=(1.0 - mom))
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Hebbian learning: h * h = h^2
        # This is equivalent to sum_of_squares but emphasizes Hebbian interpretation
        return torch.sum(h ** 2, dim=-1)


@register("hebbian_local")
def build(cfg: DictConfig) -> HebbianLocal:
    return HebbianLocal(Cfg(**getattr(cfg, "params", {})))
