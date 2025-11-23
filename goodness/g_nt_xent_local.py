from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    temperature: float = 0.1
    eps: float = 1e-8
    normalize: bool = True
    detach_negatives: bool = True
    contrastive_weight: float = 0.1


class NTXentLocal:
    """
    NT-Xent adapted for FF framework.
    Goodness = sum(h^2) + contrastive_weight * NT-Xent_score
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Clamp activations to prevent extreme values
        h = h.clamp(min=-100.0, max=100.0)
        
        # Base goodness
        base_goodness = torch.sum(h ** 2, dim=-1)
        base_goodness = torch.where(
            torch.isfinite(base_goodness),
            base_goodness,
            torch.zeros_like(base_goodness)
        )
        base_goodness = base_goodness.clamp(min=0.0, max=200000.0)
        
        # NT-Xent contrastive term with numerical stability
        x = h
        if self.cfg.normalize:
            # Compute norm and check for near-zero values
            norm = torch.norm(x, p=2, dim=1, keepdim=True)
            norm = norm.clamp(min=self.cfg.eps)
            x = x / norm
            # Check for NaN after normalization
            if not torch.all(torch.isfinite(x)):
                return base_goodness
        
        # Clamp normalized features
        x = x.clamp(min=-10.0, max=10.0)
        
        sim = x @ (x.t().detach() if self.cfg.detach_negatives else x.t())
        
        # Check for NaN in similarity matrix
        if not torch.all(torch.isfinite(sim)):
            return base_goodness
        
        # Clamp similarity values
        sim = sim.clamp(min=-100.0, max=100.0)
        sim = sim - sim.max(dim=1, keepdim=True).values
        
        logits = sim / max(self.cfg.temperature, self.cfg.eps)
        logits = logits.clamp(min=-88.0, max=88.0)
        
        try:
            lse = torch.logsumexp(logits, dim=1)
            if not torch.all(torch.isfinite(lse)):
                return base_goodness
        except:
            return base_goodness
        
        pos = torch.diagonal(logits)
        contrastive_score = pos - lse
        
        # Check contrastive score is finite
        if not torch.all(torch.isfinite(contrastive_score)):
            return base_goodness
        
        # Clamp contrastive score
        contrastive_score = contrastive_score.clamp(min=-100.0, max=100.0)
        
        # Combine with safety checks
        base_mean = base_goodness.mean()
        if not torch.isfinite(base_mean) or base_mean < self.cfg.eps:
            base_mean = torch.tensor(1.0, device=base_goodness.device, dtype=base_goodness.dtype)
        
        goodness = base_goodness + self.cfg.contrastive_weight * base_mean * contrastive_score
        goodness = goodness.clamp(min=0.0, max=200000.0)
        goodness = torch.where(torch.isfinite(goodness), goodness, base_goodness)
        
        return goodness


@register("nt_xent_local")
def build(cfg: DictConfig) -> NTXentLocal:
    return NTXentLocal(Cfg(**getattr(cfg, "params", {})))
