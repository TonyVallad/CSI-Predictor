"""
Training package for CSI-Predictor.

This package contains all training-related functionality including:
- Main training logic
- Optimizer management
- Learning rate scheduling
- Training callbacks
"""

from .trainer import train_model
from .loss import WeightedCSILoss
from .metrics import compute_f1_metrics, compute_precision_recall, compute_csi_average_metrics, compute_ahf_classification_metrics
from .optimizer import create_optimizer, create_scheduler
from .callbacks import EarlyStopping, MetricsTracker

__all__ = [
    'train_model',
    'WeightedCSILoss',
    'compute_f1_metrics',
    'compute_precision_recall',
    'compute_csi_average_metrics',
    'compute_ahf_classification_metrics',
    'create_optimizer',
    'create_scheduler',
    'EarlyStopping',
    'MetricsTracker'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 