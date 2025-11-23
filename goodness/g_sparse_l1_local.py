from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    lam: float = 0.01  # L1 penalty weight (small to not dominate)


class SparseL1Local:
    """
    L1 sparsity penalty over activations per sample.
    Goodness = sum(h^2) - lam * sum(|h|)
    Uses sum of squares as base with L1 penalty for sparsity.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Use sum of squares as base goodness, subtract L1 penalty for sparsity
        sum_squares = torch.sum(h ** 2, dim=-1)
        l1_penalty = h.abs().sum(dim=1)
        return sum_squares - self.cfg.lam * l1_penalty


@register("sparse_l1_local")
def build(cfg: DictConfig) -> SparseL1Local:
    return SparseL1Local(Cfg(**getattr(cfg, "params", {})))
