"""
Visualization package for CSI-Predictor.

This package contains all visualization functionality including:
- Plotting utilities
- Confusion matrix plots
- Training curve plots
"""

from .confusion_matrix import save_confusion_matrix_graphs, create_confusion_matrix_grid, create_overall_confusion_matrix
from .plots import plot_training_curves, create_roc_curves, create_precision_recall_curves

__all__ = [
    'save_confusion_matrix_graphs',
    'create_confusion_matrix_grid',
    'create_overall_confusion_matrix',
    'plot_training_curves',
    'create_roc_curves',
    'create_precision_recall_curves'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 