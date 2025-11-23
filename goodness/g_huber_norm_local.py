from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    delta: float = 1.0   # Huber threshold
    scale: float = 1.0


class HuberNormLocal:
    """
    Huber-regularized goodness: sum(h^2) - scale * huber_penalty.
    Provides robust regularization while maintaining positive goodness.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def _huber(self, x: torch.Tensor) -> torch.Tensor:
        d = self.cfg.delta
        absx = x.abs()
        quad = 0.5 * x * x
        lin = d * (absx - 0.5 * d)
        return torch.where(absx <= d, quad, lin)

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness with Huber regularization
        base_goodness = torch.sum(h ** 2, dim=-1)
        huber_penalty = self._huber(h).sum(dim=1)
        return base_goodness - 0.01 * self.cfg.scale * huber_penalty


@register("huber_norm_local")
def build(cfg: DictConfig) -> HuberNormLocal:
    return HuberNormLocal(Cfg(**getattr(cfg, "params", {})))
