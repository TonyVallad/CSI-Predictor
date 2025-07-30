"""
Confusion matrix utilities for CSI-Predictor.

This module contains confusion matrix functionality extracted from the original src/metrics.py file.
"""

import torch
from typing import Dict, List, Optional

def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Compute confusion matrix using PyTorch.
    
    Args:
        predictions: Predicted class indices [N]
        targets: Ground truth class indices [N]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    # Create confusion matrix using bincount
    indices = num_classes * targets + predictions
    cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm.float()

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 