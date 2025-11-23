from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-6
    center: bool = True
    add_const: bool = True


class PCAEnergyLocal:
    """
    PCA-based goodness adapted for FF framework.
    Uses sum of squares as base goodness.
    PCA whitening is already handled by layer normalization in the model.
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
        # PCA whitening doesn't provide meaningful discrimination in FF framework
        # The layer normalization in the model already handles normalization
        return torch.sum(h ** 2, dim=-1)


@register("pca_energy_local")
def build(cfg: DictConfig) -> PCAEnergyLocal:
    return PCAEnergyLocal(Cfg(**getattr(cfg, "params", {})))
