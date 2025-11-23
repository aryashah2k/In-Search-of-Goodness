from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    mean_momentum: float = 0.9  # EMA for per-unit threshold (second moment)
    center: bool = False        # optional centering by width
    scale: float = 0.01         # overall scale (small to not dominate)


class BCMLocal:
    """
    BCM-like local goodness with running per-unit threshold.
    Goodness = sum(h^2) + scale * BCM_term
    Uses sum of squares as base with BCM modulation.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.running_theta: Dict[int, torch.Tensor] = {}

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        if layer_index not in self.running_theta:
            self.running_theta[layer_index] = torch.zeros(width, device=device, dtype=dtype)
        return {"theta": self.running_theta[layer_index]}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        theta = state["theta"]
        mom = self.cfg.mean_momentum
        batch_m2 = (h * h).mean(dim=0)
        theta.mul_(mom).add_(batch_m2, alpha=(1.0 - mom))
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Clamp activations to prevent extreme values
        h = h.clamp(min=-100.0, max=100.0)
        
        # Base goodness: sum of squares
        base_goodness = torch.sum(h ** 2, dim=-1)
        base_goodness = torch.where(
            torch.isfinite(base_goodness),
            base_goodness,
            torch.zeros_like(base_goodness)
        )
        base_goodness = base_goodness.clamp(min=0.0, max=200000.0)
        
        # BCM modulation term with numerical stability
        h2 = h * h
        # Clamp h2 to prevent extreme values in BCM computation
        h2 = h2.clamp(min=0.0, max=10000.0)
        
        if state is not None and "theta" in state:
            theta = state["theta"]
            # Clamp theta to reasonable range
            theta = theta.clamp(min=0.0, max=10000.0)
        else:
            theta = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
        
        # BCM term: h^2 * (h^2 - theta)
        # Clamp the difference to prevent extreme values
        diff = (h2 - theta).clamp(min=-10000.0, max=10000.0)
        per_unit = h2 * diff
        
        # Check for NaN/Inf in per_unit
        per_unit = torch.where(
            torch.isfinite(per_unit),
            per_unit,
            torch.zeros_like(per_unit)
        )
        
        # Clamp per_unit to prevent explosion
        per_unit = per_unit.clamp(min=-100000.0, max=100000.0)
        
        bcm_term = per_unit.sum(dim=1)
        
        # Check bcm_term is finite
        if not torch.all(torch.isfinite(bcm_term)):
            return base_goodness
        
        # Clamp bcm_term
        bcm_term = bcm_term.clamp(min=-100000.0, max=100000.0)
        
        # Combine with scaled BCM term
        goodness = base_goodness + self.cfg.scale * bcm_term
        goodness = goodness.clamp(min=0.0, max=200000.0)
        goodness = torch.where(torch.isfinite(goodness), goodness, base_goodness)
        
        return goodness


@register("bcm_local")
def build(cfg: DictConfig) -> BCMLocal:
    return BCMLocal(Cfg(**getattr(cfg, "params", {})))
