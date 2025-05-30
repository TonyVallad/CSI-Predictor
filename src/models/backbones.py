"""
Backbone architectures for feature extraction in CSI-Predictor.
Supports various CNN and transformer architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any


class ResNetBackbone(nn.Module):
    """ResNet backbone for feature extraction."""
    
    def __init__(self, architecture: str = "resnet50", pretrained: bool = True):
        """
        Initialize ResNet backbone.
        
        Args:
            architecture: ResNet variant (resnet18, resnet34, resnet50, resnet101)
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        # Get ResNet model
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
        elif architecture == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet architecture: {architecture}")
        
        # Remove final classification layer
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Set feature dimension
        if architecture in ["resnet18", "resnet34"]:
            self.feature_dim = 512
        else:
            self.feature_dim = 2048
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return features


class EfficientNetBackbone(nn.Module):
    """EfficientNet backbone for feature extraction."""
    
    def __init__(self, architecture: str = "efficientnet_b0", pretrained: bool = True):
        """
        Initialize EfficientNet backbone.
        
        Args:
            architecture: EfficientNet variant
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        # Get EfficientNet model
        if architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        elif architecture == "efficientnet_b1":
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            self.feature_dim = 1280
        elif architecture == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            self.feature_dim = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet architecture: {architecture}")
        
        # Remove final classification layer
        self.feature_extractor = self.backbone.features
        self.avgpool = self.backbone.avgpool
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return features


class DenseNetBackbone(nn.Module):
    """DenseNet backbone for feature extraction."""
    
    def __init__(self, architecture: str = "densenet121", pretrained: bool = True):
        """
        Initialize DenseNet backbone.
        
        Args:
            architecture: DenseNet variant
            pretrained: Use pretrained weights
        """
        super().__init__()
        
        # Get DenseNet model
        if architecture == "densenet121":
            self.backbone = models.densenet121(pretrained=pretrained)
            self.feature_dim = 1024
        elif architecture == "densenet169":
            self.backbone = models.densenet169(pretrained=pretrained)
            self.feature_dim = 1664
        elif architecture == "densenet201":
            self.backbone = models.densenet201(pretrained=pretrained)
            self.feature_dim = 1920
        else:
            raise ValueError(f"Unsupported DenseNet architecture: {architecture}")
        
        # Remove final classification layer
        self.feature_extractor = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        features = self.feature_extractor(x)
        features = nn.functional.relu(features, inplace=True)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return features


def get_backbone(architecture: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to get backbone architecture.
    
    Args:
        architecture: Backbone architecture name
        pretrained: Use pretrained weights
        
    Returns:
        Backbone model
    """
    architecture = architecture.lower()
    
    if architecture.startswith("resnet"):
        return ResNetBackbone(architecture, pretrained)
    elif architecture.startswith("efficientnet"):
        return EfficientNetBackbone(architecture, pretrained)
    elif architecture.startswith("densenet"):
        return DenseNetBackbone(architecture, pretrained)
    else:
        raise ValueError(f"Unsupported backbone architecture: {architecture}")


def get_backbone_feature_dim(architecture: str) -> int:
    """
    Get feature dimension for backbone architecture.
    
    Args:
        architecture: Backbone architecture name
        
    Returns:
        Feature dimension
    """
    # Create temporary backbone to get feature dimension
    backbone = get_backbone(architecture, pretrained=False)
    return backbone.feature_dim 