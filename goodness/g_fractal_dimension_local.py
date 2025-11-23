from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    complexity_weight: float = 0.1  # Weight for complexity component
    eps: float = 1e-8


class FractalDimensionLocal:
    """
    Complexity-based goodness inspired by fractal dimension.
    Measures the intrinsic dimensionality/complexity of activation patterns.
    
    Goodness = sum(h²) * (1 + w * complexity_score)
    
    Uses efficient variance-based complexity measure.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def _estimate_complexity(self, h: torch.Tensor) -> torch.Tensor:
        """
        Estimate activation complexity using variance and entropy.
        Higher complexity indicates richer, more diverse representations.
        
        Args:
            h: Activations [batch_size, width]
            
        Returns:
            Complexity scores [batch_size]
        """
        # Normalize activations per sample
        h_abs = torch.abs(h)
        h_normalized = h_abs / (h_abs.sum(dim=1, keepdim=True) + self.cfg.eps)
        
        # Compute entropy as complexity measure
        # Higher entropy = more distributed activations = higher complexity
        entropy = -(h_normalized * torch.log(h_normalized + self.cfg.eps)).sum(dim=1)
        
        # Normalize by max possible entropy (log(D))
        D = h.shape[1]
        max_entropy = torch.log(torch.tensor(D, dtype=h.dtype, device=h.device))
        normalized_entropy = entropy / (max_entropy + self.cfg.eps)
        
        return normalized_entropy

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness: sum of squares
        base_goodness = torch.sum(h ** 2, dim=-1)
        
        # Estimate complexity (detached for efficiency)
        with torch.no_grad():
            complexity = self._estimate_complexity(h)
        
        # Goodness increases with complexity (more diverse = better)
        goodness = base_goodness * (1 + self.cfg.complexity_weight * complexity)
        
        return goodness


@register("fractal_dimension_local")
def build(cfg: DictConfig) -> FractalDimensionLocal:
    return FractalDimensionLocal(Cfg(**getattr(cfg, "params", {})))
