from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    temperature: float = 1.0  # tau, controls quadratic scale
    center: bool = True       # add width/(2*tau) for stability


class TemperedEnergyLocal:
    """
    Tempered goodness adapted for FF.
    Goodness = sum(h^2) / tau
    Temperature parameter scales the goodness magnitude.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        tau = max(self.cfg.temperature, self.cfg.eps)
        coef = 1.0 / tau
        quad = (h * h).sum(dim=1)
        return coef * quad


@register("tempered_energy_local")
def build(cfg: DictConfig) -> TemperedEnergyLocal:
    return TemperedEnergyLocal(Cfg(**getattr(cfg, "params", {})))
