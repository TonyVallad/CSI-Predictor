"""
Feature extraction backbones for CSI-Predictor.
Supports multiple architectures including ResNet, CheXNet, custom CNNs, and RadDINO.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights
from loguru import logger
from typing import Dict, Any

# Conditional import for RadDINO (requires transformers library)
try:
    import transformers
    from transformers import AutoModel, AutoImageProcessor
    from .rad_dino import RadDINOBackboneOnly
    RADDINO_AVAILABLE = True
    logger.debug("RadDINO backbone is available (transformers library found)")
except ImportError as e:
    logger.warning(f"RadDINO backbone not available. Missing dependency: {e}")
    RADDINO_AVAILABLE = False
    RadDINOBackboneOnly = None


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


class CheXNetBackbone(nn.Module):
    """CheXNet: DenseNet121 adapted for chest X-rays."""
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize CheXNet backbone.
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        if pretrained:
            weights = DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        # Load DenseNet121
        self.densenet = models.densenet121(weights=weights)
        
        # Modify first conv layer for chest X-rays (keep 3 channels for compatibility)
        # In a real implementation, you might want to change this to 1 channel
        original_conv = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Copy weights if pretrained
        if pretrained:
            with torch.no_grad():
                # Average the weights across input channels for grayscale compatibility
                self.densenet.features.conv0.weight.data = original_conv.weight.data
        
        # Remove classifier to use as feature extractor
        self.features = self.densenet.features
        self.feature_dim = 1024  # DenseNet121 output features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        features = self.features(x)
        # Global average pooling
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        return features


class ResNet50Backbone(nn.Module):
    """ResNet50 backbone for feature extraction."""
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize ResNet50 backbone.
        
        Args:
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()
        
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None
            
        # Load ResNet50
        resnet = models.resnet50(weights=weights)
        
        # Remove final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048  # ResNet50 output features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        features = self.features(x)
        features = torch.flatten(features, 1)
        return features


class RadDINOBackbone(nn.Module):
    """
    RadDINO backbone: Microsoft's Rad-DINO model for chest radiography.
    Based on Vision Transformer architecture specifically trained on chest X-rays.
    """
    
    def __init__(self, pretrained: bool = True):
        """
        Initialize RadDINO backbone.
        
        Args:
            pretrained: Whether to use pretrained weights (always True for RadDINO)
        """
        super().__init__()
        
        if not RADDINO_AVAILABLE:
            raise ImportError(
                "RadDINO backbone requires the transformers library. "
                "Install it with: pip install transformers>=4.30.0"
            )
        
        # Use the dedicated RadDINO implementation
        self.backbone = RadDINOBackboneOnly(pretrained=pretrained)
        self.feature_dim = self.backbone.feature_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RadDINO backbone.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        return self.backbone(x)


def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create backbone networks.
    
    Args:
        name: Backbone architecture name
        pretrained: Whether to use pretrained weights
        
    Returns:
        Backbone network module
        
    Raises:
        ValueError: If backbone name is not supported
    """
    name = name.lower().replace('_', '').replace('-', '')
    
    if name == "resnet50":
        logger.info(f"Creating ResNet50 backbone (pretrained={pretrained})")
        return ResNet50Backbone(pretrained=pretrained)
    
    elif name == "chexnet":
        logger.info(f"Creating CheXNet (DenseNet121) backbone (pretrained={pretrained})")
        return CheXNetBackbone(pretrained=pretrained)
    
    elif name == "custom1":
        logger.info("Creating Custom 5-layer CNN backbone")
        return CustomCNNBackbone(input_channels=3)
    
    elif name == "raddino":
        if not RADDINO_AVAILABLE:
            raise ImportError(
                "RadDINO backbone requires the transformers library. "
                "Install it with: pip install transformers>=4.30.0"
            )
        logger.info(f"Creating RadDINO backbone (pretrained={pretrained})")
        return RadDINOBackbone(pretrained=pretrained)
    
    else:
        available_backbones = ["ResNet50", "CheXNet", "Custom_1"]
        if RADDINO_AVAILABLE:
            available_backbones.append("RadDINO")
        raise ValueError(f"Unsupported backbone: {name}. Available: {available_backbones}")


def get_backbone_feature_dim(name: str) -> int:
    """
    Get the output feature dimension for a backbone.
    
    Args:
        name: Backbone architecture name
        
    Returns:
        Feature dimension
    """
    name = name.lower().replace('_', '').replace('-', '')
    
    backbone_dims = {
        "resnet50": 2048,
        "chexnet": 1024,
        "custom1": 1024,
        "raddino": 768  # RadDINO feature dimension
    }
    
    if name == "raddino" and not RADDINO_AVAILABLE:
        raise ImportError(
            "RadDINO backbone requires the transformers library. "
            "Install it with: pip install transformers>=4.30.0"
        )
    
    if name in backbone_dims:
        return backbone_dims[name]
    else:
        available_backbones = ["ResNet50", "CheXNet", "Custom_1"]
        if RADDINO_AVAILABLE:
            available_backbones.append("RadDINO")
        raise ValueError(f"Unknown backbone: {name}. Available: {available_backbones}")


# Export main functions
__all__ = [
    'get_backbone',
    'get_backbone_feature_dim',
    'ResNet50Backbone',
    'CheXNetBackbone',
    'CustomCNNBackbone',
    'RadDINOBackbone'
] 