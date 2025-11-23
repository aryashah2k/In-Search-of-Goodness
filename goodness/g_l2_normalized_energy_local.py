from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    add_const: bool = False


class L2NormalizedEnergyLocal:
    """
    L2-normalized energy adapted for FF.
    Goodness = sum(h^2) (standard sum of squares)
    The L2 normalization is applied during peer normalization in the model.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Use standard sum of squares
        # L2 normalization doesn't provide meaningful discrimination in FF framework
        return torch.sum(h ** 2, dim=-1)


@register("l2_normalized_energy_local")
def build(cfg: DictConfig) -> L2NormalizedEnergyLocal:
    return L2NormalizedEnergyLocal(Cfg(**getattr(cfg, "params", {})))
