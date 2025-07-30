"""
Loss functions for CSI-Predictor training.

This module contains loss function implementations extracted from the original src/train.py file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

class WeightedCSILoss(nn.Module):
    """
    Weighted Cross-Entropy Loss that reduces importance of unknown class
    but still learns to predict it.
    
    This treats "ungradable" or "unknown" CSI zones (class 4) as a valid 
    prediction target rather than ignoring it, but gives it reduced weight
    to emphasize learning the clear CSI classifications (0-3).
    """
    
    def __init__(self, unknown_weight: float = 0.3):
        """
        Initialize weighted cross-entropy loss.
        
        Args:
            unknown_weight: Weight for unknown class (default: 0.3)
                          - 1.0 = equal importance with other classes
                          - 0.5 = half importance  
                          - 0.1 = very low importance
        """
        super().__init__()
        self.unknown_weight = unknown_weight
        
        # Weights: [Normal, Mild, Moderate, Severe, Unknown]
        # Classes 0-3 get full weight, class 4 gets reduced weight
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, unknown_weight])
        
        # Register weights as buffer so they move to device automatically
        self.register_buffer('class_weights', weights)
        
        # Initialize CrossEntropyLoss without weights initially
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        
        logger.info(f"Initialized WeightedCSILoss with unknown_weight={unknown_weight}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.
        
        Args:
            predictions: Model predictions [batch_size, n_zones, n_classes]
            targets: Ground truth labels [batch_size, n_zones]
            
        Returns:
            Scalar loss value
        """
        batch_size, n_zones, n_classes = predictions.shape
        
        # Reshape for cross-entropy: [batch_size * n_zones, n_classes]
        predictions_flat = predictions.view(-1, n_classes)
        targets_flat = targets.view(-1)
        
        # Compute weighted cross-entropy using F.cross_entropy with weights
        return F.cross_entropy(predictions_flat, targets_flat, weight=self.class_weights)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 