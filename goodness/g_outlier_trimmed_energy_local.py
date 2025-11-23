from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    trim_frac: float = 0.1  # fraction to trim from each tail along units
    center: bool = False


class OutlierTrimmedEnergyLocal:
    """
    Outlier-trimmed sum of squares:
      Sort per-sample squared activations, keep middle values (trim outliers).
      Returns positive goodness for FF framework.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        B, W = h.shape
        k = int(max(0, min(W // 2, round(self.cfg.trim_frac * W))))
        h2 = (h * h)
        # sort per row
        sorted_h2, _ = torch.sort(h2, dim=1)
        if k > 0:
            mid = sorted_h2[:, k: W - k]
        else:
            mid = sorted_h2
        # Return positive goodness (sum of trimmed squares)
        return mid.sum(dim=1)


@register("outlier_trimmed_energy_local")
def build(cfg: DictConfig) -> OutlierTrimmedEnergyLocal:
    return OutlierTrimmedEnergyLocal(Cfg(**getattr(cfg, "params", {})))
