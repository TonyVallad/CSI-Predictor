"""
Model heads package for CSI-Predictor.

This package contains all classification and regression head implementations including:
- CSI classification head
- Regression head for backward compatibility
"""

from .csi_head import CSIHead
from .regression_head import CSIRegressionHead

__all__ = [
    'CSIHead',
    'CSIRegressionHead'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 