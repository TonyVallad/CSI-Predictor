"""
Metrics package for CSI-Predictor.

This package contains all metrics computation functionality including:
- Classification metrics
- Confusion matrix utilities
- F1 score calculations
"""

from .evaluation_metrics import compute_confusion_matrices_per_zone, create_classification_report_per_zone
from .classification import compute_classification_metrics
from .f1_score import compute_f1_score, compute_f1_scores_per_class
from .confusion_matrix import create_confusion_matrix

__all__ = [
    'compute_confusion_matrices_per_zone',
    'create_classification_report_per_zone',
    'compute_classification_metrics',
    'compute_f1_score',
    'compute_f1_scores_per_class',
    'create_confusion_matrix'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 