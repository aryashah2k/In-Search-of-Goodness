from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    sigma2: float = 1.0   # variance for Gaussian energy
    center: bool = True   # add width term for stability


class GaussianEnergyLocal:
    """
    Gaussian-like local energy over activations.
    Modified to return positive goodness: sum(h^2) / sigma2
    Higher activations = higher goodness (inverted from typical energy formulation).
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Invert energy formulation: higher activations = higher goodness
        quad = (h * h).sum(dim=1)
        coef = 1.0 / max(self.cfg.sigma2, self.cfg.eps)
        return coef * quad


@register("gaussian_energy_local")
def build(cfg: DictConfig) -> GaussianEnergyLocal:
    return GaussianEnergyLocal(Cfg(**getattr(cfg, "params", {})))
