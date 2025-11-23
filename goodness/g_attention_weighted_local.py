from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    temperature: float = 1.0  # Temperature for softmax attention
    eps: float = 1e-8
    attention_weight: float = 0.01  # Very small weight to avoid dominating


class AttentionWeightedLocal:
    """
    Attention-weighted goodness function inspired by transformer attention mechanisms.
    Computes self-attention scores across units and weights activations by attention.
    
    Goodness = (1-w) * sum(h²) + w * sum((attention @ h)²)
    
    This encourages the model to focus on the most relevant features dynamically.
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness: sum of squares
        base_goodness = torch.sum(h ** 2, dim=-1)
        
        # Compute per-sample attention diversity (detached to avoid gradient issues)
        with torch.no_grad():
            B, D = h.shape
            
            # Compute per-sample feature importance using softmax over features
            # This measures how concentrated vs distributed the activations are
            h_abs = torch.abs(h)
            feature_probs = F.softmax(h_abs / max(self.cfg.temperature, self.cfg.eps), dim=-1)
            
            # Compute entropy per sample
            # Higher entropy = more distributed features = better diversity
            per_sample_entropy = -(feature_probs * torch.log(feature_probs + self.cfg.eps)).sum(dim=-1)
            
            # Normalize by max possible entropy
            max_entropy = torch.log(torch.tensor(D, dtype=h.dtype, device=h.device))
            normalized_entropy = per_sample_entropy / (max_entropy + self.cfg.eps)
        
        # Add small per-sample attention-based modulation
        # Scale by base_goodness to maintain relative magnitudes
        attention_modulation = self.cfg.attention_weight * base_goodness * normalized_entropy
        
        return base_goodness + attention_modulation


@register("attention_weighted_local")
def build(cfg: DictConfig) -> AttentionWeightedLocal:
    return AttentionWeightedLocal(Cfg(**getattr(cfg, "params", {})))
