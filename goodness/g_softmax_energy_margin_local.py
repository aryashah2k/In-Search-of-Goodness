from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    eps: float = 1e-8
    temperature: float = 1.0  # softmax temperature
    margin: float = 0.0       # optional margin added to goodness
    scale: float = 0.1        # overall scale (small weight)


class SoftmaxEnergyMarginLocal:
    """
    Softmax-energy with margin adapted for FF.
    Goodness = sum(h^2) + scale * softmax_entropy_term
    Encourages peaky distributions while maintaining positive goodness.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness: sum of squares (same as baseline)
        # This naturally scales with hidden_dim (e.g., 2000)
        base_goodness = torch.sum(h ** 2, dim=-1)
        
        # Replace NaN/Inf with zeros
        base_goodness = torch.where(
            torch.isfinite(base_goodness),
            base_goodness,
            torch.zeros_like(base_goodness)
        )
        
        # Clamp extreme values but keep scale appropriate for threshold comparison
        # Max reasonable value: hidden_dim * max_activation^2 ≈ 2000 * 100 = 200000
        base_goodness = base_goodness.clamp(min=0.0, max=200000.0)
        
        # Softmax entropy term (negative entropy encourages peaky distributions)
        tau = max(self.cfg.temperature, self.cfg.eps)
        
        # Clamp h to prevent overflow but allow reasonable range
        h_safe = h.clamp(min=-100.0, max=100.0)
        
        # Check for NaN in h
        if not torch.all(torch.isfinite(h_safe)):
            return base_goodness
        
        logits = h_safe / tau
        
        # Numerical stability: subtract max before logsumexp
        row_max = logits.max(dim=1, keepdim=True).values
        if not torch.all(torch.isfinite(row_max)):
            return base_goodness
            
        logits = logits - row_max
        
        # Clamp logits to prevent overflow in exp
        logits = logits.clamp(min=-88.0, max=88.0)  # exp(88) is near float32 max
        
        # Safe logsumexp
        try:
            lse = torch.logsumexp(logits, dim=1)
            if not torch.all(torch.isfinite(lse)):
                return base_goodness
        except:
            return base_goodness
            
        entropy_term = -lse  # negative logsumexp
        
        # Clamp entropy term to reasonable range
        entropy_term = entropy_term.clamp(min=-100.0, max=100.0)
        
        if not torch.all(torch.isfinite(entropy_term)):
            return base_goodness
        
        # Compute final goodness with numerical stability
        base_mean = base_goodness.mean()
        if not torch.isfinite(base_mean) or base_mean < self.cfg.eps:
            base_mean = torch.tensor(1.0, device=base_goodness.device, dtype=base_goodness.dtype)
        
        # Compute scaled entropy contribution
        # Keep scale small so it modulates but doesn't dominate base_goodness
        entropy_contribution = self.cfg.scale * base_mean * entropy_term
        
        if not torch.all(torch.isfinite(entropy_contribution)):
            return base_goodness
        
        goodness = base_goodness + entropy_contribution + self.cfg.margin
        
        # Clamp final goodness to prevent extreme values
        # Keep upper bound high enough to allow discrimination around threshold (2000)
        goodness = goodness.clamp(min=0.0, max=200000.0)
        
        # Final safety check
        goodness = torch.where(torch.isfinite(goodness), goodness, base_goodness)
        
        return goodness


@register("softmax_energy_margin_local")
def build(cfg: DictConfig) -> SoftmaxEnergyMarginLocal:
    return SoftmaxEnergyMarginLocal(Cfg(**getattr(cfg, "params", {})))
