from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from omegaconf import DictConfig

from .registry import register


@dataclass
class Cfg:
    prediction_momentum: float = 0.9  # EMA momentum for prediction
    prediction_weight: float = 0.1  # Weight for prediction error term
    eps: float = 1e-8


class PredictiveCodingLocal:
    """
    Predictive coding goodness inspired by predictive processing theory.
    
    The brain minimizes prediction errors by maintaining predictions of expected
    activity and updating them based on actual observations.
    
    Goodness = sum(h²) - λ * ||h - μ_predicted||²
    
    where μ_predicted is an exponential moving average of past activations,
    representing learned expectations/predictions.
    
    Higher goodness when activations match predictions (low surprise).
    """

    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.predictions: Dict[int, torch.Tensor] = {}

    def init_state(self, layer_index: int, width: int, device: torch.device, dtype: torch.dtype) -> Dict:
        """Initialize prediction state for this layer."""
        if layer_index not in self.predictions:
            # Initialize predictions to zero (no prior expectations)
            self.predictions[layer_index] = torch.zeros(width, device=device, dtype=dtype)
        return {"prediction": self.predictions[layer_index]}

    @torch.no_grad()
    def update_state(self, state: Dict, h: torch.Tensor) -> Dict:
        """
        Update prediction using exponential moving average.
        
        prediction_new = momentum * prediction_old + (1 - momentum) * h_mean
        """
        prediction = state["prediction"]
        momentum = self.cfg.prediction_momentum
        
        # Update prediction with batch mean
        batch_mean = h.mean(dim=0)
        prediction.mul_(momentum).add_(batch_mean, alpha=(1.0 - momentum))
        
        return state

    def goodness(self, h: torch.Tensor, layer_index: int, state: Optional[Dict]) -> torch.Tensor:
        # Base goodness: sum of squares
        base_goodness = torch.sum(h ** 2, dim=-1)
        
        # Get prediction (expected activation pattern)
        if state is not None and "prediction" in state:
            prediction = state["prediction"]
        else:
            # Fallback: no prediction available, use zeros
            prediction = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype)
        
        # Compute prediction error: ||h - prediction||²
        prediction_error = torch.sum((h - prediction) ** 2, dim=-1)
        
        # Scale prediction error with proper numerical stability
        # Avoid division by very small numbers
        base_mean = base_goodness.mean().clamp(min=self.cfg.eps)
        base_clamped = base_goodness.clamp(min=self.cfg.eps)
        
        # Normalize prediction error relative to base goodness
        error_ratio = prediction_error / base_clamped
        # Clamp to prevent extreme values
        error_ratio = error_ratio.clamp(max=10.0)
        
        scaled_error = self.cfg.prediction_weight * base_mean * error_ratio
        
        # Goodness decreases with prediction error (we want predictable activations)
        goodness = base_goodness - scaled_error
        
        # Final safety check: replace any NaN or Inf with base goodness
        goodness = torch.where(torch.isfinite(goodness), goodness, base_goodness)
        
        return goodness


@register("predictive_coding_local")
def build(cfg: DictConfig) -> PredictiveCodingLocal:
    return PredictiveCodingLocal(Cfg(**getattr(cfg, "params", {})))
