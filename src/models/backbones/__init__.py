"""
Model backbones package for CSI-Predictor.

This package contains all backbone model implementations including:
- Base backbone classes
- ResNet backbones
- DenseNet/CheXNet backbones
- Custom CNN backbone
- RadDINO backbone
"""

from .resnet import ResNet50Backbone
from .densenet import CheXNetBackbone
from .custom import CustomCNNBackbone
from .raddino import RadDINOBackbone

def get_backbone(backbone_arch: str, pretrained: bool = True):
    """
    Factory function to get backbone by architecture name.
    
    Args:
        backbone_arch: Backbone architecture name
        pretrained: Use pretrained weights
        
    Returns:
        Backbone model
    """
    backbone_map = {
        'resnet50': ResNet50Backbone,
        'densenet121': CheXNetBackbone,
        'chexnet': CheXNetBackbone,
        'custom': CustomCNNBackbone,
        'custom_cnn': CustomCNNBackbone,
        'raddino': RadDINOBackbone,
    }
    
    if backbone_arch not in backbone_map:
        raise ValueError(f"Unknown backbone architecture: {backbone_arch}. Available: {list(backbone_map.keys())}")
    
    return backbone_map[backbone_arch](pretrained=pretrained)

def get_backbone_feature_dim(backbone_arch: str) -> int:
    """
    Get feature dimension for a given backbone architecture.
    
    Args:
        backbone_arch: Backbone architecture name
        
    Returns:
        Feature dimension
    """
    feature_dims = {
        'resnet50': 2048,
        'densenet121': 1024,
        'chexnet': 1024,
        'custom': 512,
        'custom_cnn': 512,
        'raddino': 768,  # RadDINO typically uses 768
    }
    
    if backbone_arch not in feature_dims:
        raise ValueError(f"Unknown backbone architecture: {backbone_arch}. Available: {list(feature_dims.keys())}")
    
    return feature_dims[backbone_arch]

__all__ = [
    'get_backbone',
    'get_backbone_feature_dim',
    'ResNet50Backbone',
    'CheXNetBackbone',
    'CustomCNNBackbone',
    'RadDINOBackbone'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 