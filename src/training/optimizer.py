"""
Optimizer management for CSI-Predictor training.

This module contains optimizer functionality extracted from the original src/train.py file.
"""

import torch
import torch.optim as optim
from typing import Dict, Any
from ..config import Config

def create_optimizer(model: torch.nn.Module, config: Config) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Args:
        model: Model to optimize
        config: Configuration object
        
    Returns:
        Optimizer instance
    """
    optimizer_name = config.optimizer.lower()
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config.learning_rate, 
            momentum=0.9, 
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Config) -> optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration object
        
    Returns:
        Learning rate scheduler
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.5, 
        verbose=True
    )
    
    return scheduler


def get_learning_rate(optimizer: optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer instance
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 