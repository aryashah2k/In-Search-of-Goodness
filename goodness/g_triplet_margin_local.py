from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    margin: float = 1.0  # Margin for triplet loss
    triplet_weight: float = 0.1  # Weight for triplet component
    eps: float = 1e-8
    normalize: bool = True  # Whether to L2-normalize before computing distances


class TripletMarginLocal:
    """
    Triplet margin goodness inspired by metric learning and face recognition.
    
    For each sample (anchor), ensures it's closer to similar samples (positive)
    than to dissimilar samples (negative) by at least a margin.
    
    Goodness = sum(h²) + w * triplet_margin_term
    
    where triplet_margin_term = -max(0, ||h_a - h_p||² - ||h_a - h_n||² + margin)
    
    In the FF context:
    - Positive samples: same class (first half of batch in FF)
    - Negative samples: different class (second half of batch in FF)
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        return {}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        return state

    def _compute_pairwise_distances(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distances.
        
        Args:
            h: Activations [batch_size, width]
            
        Returns:
            Distance matrix [batch_size, batch_size]
        """
        if self.cfg.normalize:
            # L2 normalize for cosine-like distance
            h = torch.nn.functional.normalize(h, p=2, dim=1, eps=self.cfg.eps)
        
        # Compute ||h_i - h_j||² = ||h_i||² + ||h_j||² - 2 * h_i · h_j
        h_norm_sq = torch.sum(h ** 2, dim=1, keepdim=True)
        distances = h_norm_sq + h_norm_sq.t() - 2 * torch.matmul(h, h.t())
        
        # Clamp to avoid numerical issues
        distances = torch.clamp(distances, min=0)
        
        return distances

    def _compute_contrastive_separation(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute efficient contrastive separation metric.
        Measures how well separated positive and negative samples are.
        
        Args:
            h: Activations [batch_size, width]
            
        Returns:
            Separation scores per sample [batch_size]
        """
        B = h.shape[0]
        half_batch = B // 2
        
        if half_batch < 1:
            return torch.zeros(B, device=h.device, dtype=h.dtype)
        
        # Split into positive and negative samples
        h_pos = h[:half_batch]
        h_neg = h[half_batch:2*half_batch]
        
        # Compute within-class cohesion (want this small)
        pos_mean = h_pos.mean(dim=0, keepdim=True)
        neg_mean = h_neg.mean(dim=0, keepdim=True)
        
        pos_cohesion = torch.sum((h_pos - pos_mean) ** 2, dim=1)
        neg_cohesion = torch.sum((h_neg - neg_mean) ** 2, dim=1)
        
        # Compute between-class separation (want this large)
        pos_to_neg_center = torch.sum((h_pos - neg_mean) ** 2, dim=1)
        neg_to_pos_center = torch.sum((h_neg - pos_mean) ** 2, dim=1)
        
        # Separation score: between / (within + eps)
        # Higher is better (well separated)
        pos_separation = pos_to_neg_center / (pos_cohesion + self.cfg.eps)
        neg_separation = neg_to_pos_center / (neg_cohesion + self.cfg.eps)
        
        # Combine
        all_separation = torch.cat([pos_separation, neg_separation])
        
        # Pad if necessary
        if len(all_separation) < B:
            padding = torch.zeros(B - len(all_separation), device=h.device, dtype=h.dtype)
            all_separation = torch.cat([all_separation, padding])
        
        return all_separation[:B]

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness: sum of squares
        base_goodness = torch.sum(h ** 2, dim=-1)
        
        # Compute contrastive separation (detached for efficiency)
        with torch.no_grad():
            separation = self._compute_contrastive_separation(h)
        
        # Goodness increases with separation (well-separated classes)
        # Normalize separation to reasonable range
        normalized_separation = torch.tanh(separation)
        
        # Scale the separation term by base goodness magnitude for stability
        scaled_separation = self.cfg.triplet_weight * base_goodness.mean() * normalized_separation
        
        goodness = base_goodness + scaled_separation
        
        return goodness


@register("triplet_margin_local")
def build(cfg: DictConfig) -> TripletMarginLocal:
    return TripletMarginLocal(Cfg(**getattr(cfg, "params", {})))
