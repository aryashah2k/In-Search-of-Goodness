from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    """Configuration for sum of squares goodness function."""
    pass  # No parameters needed for baseline


class SumOfSquares:
    """
    Baseline goodness function from the original Forward-Forward paper.
    Goodness = sum of squared activations per sample.
    This is the default function used in the original implementation.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        """Initialize any state needed for this goodness function."""
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        """Update state based on current batch (if needed)."""
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        """
        Compute goodness as sum of squared activations.
        
        Args:
            h: Activations tensor of shape (batch_size, hidden_dim)
            layer_index: Index of the current layer
            state: Optional state dictionary
            
        Returns:
            Goodness values of shape (batch_size,)
        """
        return torch.sum(h ** 2, dim=-1)


@register("sum_of_squares")
def build(cfg: DictConfig) -> SumOfSquares:
    return SumOfSquares(Cfg(**getattr(cfg, "params", {})))
