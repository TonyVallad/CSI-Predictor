"""
Model factory for CSI-Predictor.

This module contains model factory functionality extracted from the original src/models/__init__.py file.
"""

import torch
import torch.nn as nn
from typing import Optional
from ...utils.logging import logger
from .complete import CSIModel, CSIModelWithZoneMasking

def build_model(cfg, use_zone_focus: bool = None) -> nn.Module:
    """
    Build CSI model from configuration and move to device.
    
    Args:
        cfg: Configuration object with model settings
        use_zone_focus: Whether to use zone focus model (auto-detected if None)
        
    Returns:
        CSI model on the specified device
    """
    logger.info(f"Building CSI model with architecture: {cfg.model_arch}")
    
    # Auto-detect zone focus if not specified
    if use_zone_focus is None:
        use_zone_focus = hasattr(cfg, 'zone_focus_method') or hasattr(cfg, 'use_segmentation_masking') or hasattr(cfg, 'masking_strategy')
    
    # Create appropriate model
    if use_zone_focus:
        logger.info("Creating CSI model with zone focus support")
        model = CSIModelWithZoneMasking(
            cfg=cfg,
            backbone_arch=cfg.model_arch,
            n_classes_per_zone=5,  # CSI scores: 0, 1, 2, 3, 4
            pretrained=True
        )
    else:
        logger.info("Creating standard CSI model")
        model = CSIModel(
            backbone_arch=cfg.model_arch,
            n_classes_per_zone=5,  # CSI scores: 0, 1, 2, 3, 4
            pretrained=True,
            dropout_rate=cfg.dropout_rate
        )
    
    # Move to device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created successfully:")
    logger.info(f"  - Backbone: {cfg.model_arch}")
    logger.info(f"  - Zone focus: {use_zone_focus}")
    if use_zone_focus and hasattr(cfg, 'zone_focus_method'):
        logger.info(f"  - Zone focus method: {cfg.zone_focus_method}")
    logger.info(f"  - Total parameters: {total_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Device: {device}")
    
    return model


def build_zone_focus_model(cfg) -> CSIModelWithZoneMasking:
    """
    Build CSI model with zone focus explicitly.
    
    Args:
        cfg: Configuration object with zone focus settings
        
    Returns:
        CSI model with zone focus on the specified device
    """
    return build_model(cfg, use_zone_focus=True)


def build_zone_masking_model(cfg) -> CSIModelWithZoneMasking:
    """
    Build CSI model with zone masking explicitly (legacy function).
    
    Args:
        cfg: Configuration object with zone masking settings
        
    Returns:
        CSI model with zone focus on the specified device
    """
    return build_model(cfg, use_zone_focus=True)


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': model.__class__.__name__
    }

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 