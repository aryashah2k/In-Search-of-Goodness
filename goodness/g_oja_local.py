from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    alpha: float = 0.0001  # weight on stabilization (quartic) term - small to not dominate
    center: bool = False


class OjaLocal:
    """
    Oja-inspired local goodness: quadratic activation utility with stabilization.
    Goodness per sample: sum(h^2) - alpha * sum(h^4).
    The quartic term provides gentle regularization without changing scale dramatically.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        h2 = (h * h)
        quad = h2.sum(dim=1)
        quart = (h2 * h2).sum(dim=1)
        g = quad - self.cfg.alpha * quart
        if self.cfg.center:
            width = h.shape[1]
            g = g - torch.as_tensor(width, device=h.device, dtype=h.dtype)
        return g


@register("oja_local")
def build(cfg: DictConfig) -> OjaLocal:
    return OjaLocal(Cfg(**getattr(cfg, "params", {})))
