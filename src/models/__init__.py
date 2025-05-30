"""
Models package for CSI-Predictor.
Contains neural network architectures and components.
"""

import torch
import torch.nn as nn
from loguru import logger

from .backbones import get_backbone, get_backbone_feature_dim
from .head import CSIHead


class CSIModel(nn.Module):
    """Complete CSI prediction model with backbone + head."""
    
    def __init__(self, backbone_arch: str, n_classes_per_zone: int = 5, pretrained: bool = True):
        """
        Initialize CSI model.
        
        Args:
            backbone_arch: Backbone architecture name
            n_classes_per_zone: Number of classes per zone (default: 5)
            pretrained: Use pretrained backbone
        """
        super().__init__()
        
        # Get backbone
        self.backbone = get_backbone(backbone_arch, pretrained)
        
        # Get backbone feature dimension
        backbone_out_dim = get_backbone_feature_dim(backbone_arch)
        
        # Create CSI head
        self.head = CSIHead(backbone_out_dim, n_classes_per_zone)
        
        # Store metadata
        self.backbone_arch = backbone_arch
        self.n_classes_per_zone = n_classes_per_zone
        self.pretrained = pretrained
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through complete model.
        
        Args:
            x: Input images [batch_size, channels, height, width]
            
        Returns:
            Logits [batch_size, n_zones, n_classes_per_zone]
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Predict CSI scores using head
        predictions = self.head(features)
        
        return predictions


def build_model(cfg) -> CSIModel:
    """
    Build CSI model from configuration and move to device.
    
    Args:
        cfg: Configuration object with model settings
        
    Returns:
        CSI model on the specified device
    """
    logger.info(f"Building CSI model with architecture: {cfg.model_arch}")
    
    # Create model
    model = CSIModel(
        backbone_arch=cfg.model_arch,
        n_classes_per_zone=5,  # CSI scores: 0, 1, 2, 3, 4
        pretrained=True
    )
    
    # Move to device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created successfully:")
    logger.info(f"  - Backbone: {cfg.model_arch}")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Device: {device}")
    
    return model


# Export main components
__all__ = ['CSIModel', 'build_model'] 