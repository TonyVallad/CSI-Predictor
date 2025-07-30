"""
Regression head for CSI-Predictor.

This module contains regression head functionality extracted from the original src/models/head.py file.
"""

import torch
import torch.nn as nn

class CSIRegressionHead(nn.Module):
    """
    CSI regression head for backward compatibility.
    
    Predicts continuous CSI scores for 6 zones.
    """
    
    def __init__(self, input_dim: int, num_zones: int = 6):
        """
        Initialize CSI regression head.
        
        Args:
            input_dim: Input feature dimension
            num_zones: Number of CSI zones
        """
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_zones)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.head(x)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 