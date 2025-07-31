"""
Evaluation visualization module.

This module contains visualization functions for evaluation results.
"""

from .confusion_matrix import (
    save_confusion_matrix_graphs,
    create_confusion_matrix_grid,
    create_overall_confusion_matrix
)

from .plots import (
    plot_training_curves,
    create_roc_curves,
    create_precision_recall_curves,
    create_summary_dashboard
)

__all__ = [
    'save_confusion_matrix_graphs',
    'create_confusion_matrix_grid', 
    'create_overall_confusion_matrix',
    'plot_training_curves',
    'create_roc_curves',
    'create_precision_recall_curves',
    'create_summary_dashboard'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 