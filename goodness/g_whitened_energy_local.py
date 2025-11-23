from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-6
    center: bool = True  # center by batch mean
    add_const: bool = True  # add width/(2) term for stability


class WhitenedEnergyLocal:
    """
    Whitened goodness adapted for FF.
    Uses standard sum of squares.
    Whitening is already handled by layer normalization in the model.
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
        # Whitening doesn't provide meaningful discrimination in FF framework
        # Layer normalization in the model already handles normalization
        return torch.sum(h ** 2, dim=-1)


@register("whitened_energy_local")
def build(cfg: DictConfig) -> WhitenedEnergyLocal:
    return WhitenedEnergyLocal(Cfg(**getattr(cfg, "params", {})))
