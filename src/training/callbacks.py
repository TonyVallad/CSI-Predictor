"""
Training callbacks for CSI-Predictor.

This module contains training callback functionality extracted from the original src/utils.py file.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.counter = 0
        self.best_loss = None
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from (optional)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if model is not None and self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


class MetricsTracker:
    """Utility class to track and compute average metrics during training."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = defaultdict(list)
    
    def update(self, metric_name: str, value: float) -> None:
        """
        Update a metric with a new value.
        
        Args:
            metric_name: Name of the metric
            value: New value to add
        """
        self.metrics[metric_name].append(value)
    
    def get_average(self, metric_name: str) -> float:
        """
        Get average value for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Average value
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return 0.0
        return np.mean(self.metrics[metric_name])
    
    def get_averages(self) -> Dict[str, float]:
        """
        Get average values for all metrics.
        
        Returns:
            Dictionary of metric names to average values
        """
        return {name: self.get_average(name) for name in self.metrics}
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        """
        Initialize average meter.
        
        Args:
            name: Name of the metric
            fmt: Format string for string representation
        """
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self) -> None:
        """Reset the meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with new values.
        
        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        """String representation of the meter."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 