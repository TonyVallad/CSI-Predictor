"""
ResNet backbones for CSI-Predictor.

This module contains ResNet backbone functionality extracted from the original src/models/backbones.py file.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

class ResNet50Backbone(nn.Module):
    """ResNet50 backbone for feature extraction."""
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize ResNet50 backbone.
        
        Args:
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        if pretrained:
            # Use the latest pretrained weights
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            # Use random initialization
            self.backbone = models.resnet50(weights=None)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Global average pooling and flatten
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        self.feature_dim = 2048  # ResNet50 feature dimension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        features = self.features(x)
        pooled = self.pool(features)
        flattened = self.flatten(pooled)
        return flattened

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 