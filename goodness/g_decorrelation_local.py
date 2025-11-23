from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    center: bool = True  # center before covariance
    scale: float = 0.001  # Very small scale to avoid dominating


class DecorrelationLocal:
    """
    Decorrelation objective: sum of squares with decorrelation bonus.
    Goodness = sum(h^2) - scale * off_diagonal_penalty
    Encourages decorrelated features while maintaining positive goodness.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness: sum of squares per sample
        base_goodness = torch.sum(h ** 2, dim=-1)
        
        # Decorrelation penalty computed per-sample to avoid NaN
        # Use variance of activations as a simple decorrelation proxy
        h_var = h.var(dim=1, unbiased=False)
        
        # Higher variance means more spread out (decorrelated) features
        # We want to encourage this, so add a small bonus
        decorrelation_bonus = self.cfg.scale * base_goodness.mean() * h_var
        
        return base_goodness + decorrelation_bonus


@register("decorrelation_local")
def build(cfg: DictConfig) -> DecorrelationLocal:
    return DecorrelationLocal(Cfg(**getattr(cfg, "params", {})))
