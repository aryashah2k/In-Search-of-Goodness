from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    importance_weight: float = 0.1  # Weight for feature importance component
    eps: float = 1e-8


class GameTheoreticLocal:
    """
    Game-theoretic goodness using feature importance.
    Inspired by cooperative game theory - measures each feature's contribution.
    
    Goodness = sum(h²) + w * importance_weighted_term
    
    Uses efficient gradient-based feature importance instead of full Shapley computation.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def _compute_feature_importance(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance efficiently using magnitude and variance.
        Features with high magnitude and high variance are more important.
        
        Args:
            h: Activations [batch_size, width]
            
        Returns:
            Feature importance scores [batch_size, width]
        """
        # Magnitude importance
        magnitude = torch.abs(h)
        
        # Normalize per sample
        magnitude_normalized = magnitude / (magnitude.sum(dim=1, keepdim=True) + self.cfg.eps)
        
        return magnitude_normalized

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness: sum of squares
        base_goodness = torch.sum(h ** 2, dim=-1)
        
        # Compute feature importance (detached for efficiency)
        with torch.no_grad():
            importance = self._compute_feature_importance(h)
        
        # Weight activations by their importance
        # Features with high importance contribute more
        weighted_h_squared = (h ** 2) * (1 + importance)
        importance_weighted_goodness = torch.sum(weighted_h_squared, dim=-1)
        
        # Combine base and importance-weighted goodness
        goodness = base_goodness + self.cfg.importance_weight * (importance_weighted_goodness - base_goodness)
        
        return goodness


@register("game_theoretic_local")
def build(cfg: DictConfig) -> GameTheoreticLocal:
    return GameTheoreticLocal(Cfg(**getattr(cfg, "params", {})))
