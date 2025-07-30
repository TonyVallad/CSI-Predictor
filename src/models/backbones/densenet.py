"""
DenseNet/CheXNet backbones for CSI-Predictor.

This module contains DenseNet backbone functionality extracted from the original src/models/backbones.py file.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

class CheXNetBackbone(nn.Module):
    """CheXNet: DenseNet121 adapted for chest X-rays."""
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize CheXNet backbone.
        
        Args:
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        if pretrained:
            # Use the latest pretrained weights
            self.backbone = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            # Use random initialization
            self.backbone = models.densenet121(weights=None)
        
        # Remove the final classification layer
        self.features = self.backbone.features
        
        # Global average pooling and flatten
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.feature_dim = 1024  # DenseNet121 feature dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        features = self.features(x)
        pooled = self.pool(features)
        flattened = self.flatten(pooled)
        return flattened

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 