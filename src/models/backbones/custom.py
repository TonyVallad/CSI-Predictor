"""
Custom CNN backbone for CSI-Predictor.

This module contains custom CNN backbone functionality extracted from the original src/models/backbones.py file.
"""

import torch
import torch.nn as nn

class CustomCNNBackbone(nn.Module):
    """Simple 5-layer CNN backbone as baseline."""
    
    def __init__(self, input_channels: int = 3):
        """
        Initialize Custom CNN backbone.
        
        Args:
            input_channels: Number of input channels
        """
        super().__init__()
        
        # 5-layer CNN as specified
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.feature_dim = 1024
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        return self.features(x)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 