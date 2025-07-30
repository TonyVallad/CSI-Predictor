"""
Complete models package for CSI-Predictor.

This package contains complete model implementations that combine backbones and heads including:
- Complete RadDINO model for CSI prediction
- CSI models with and without zone masking
"""

from .csi_models import CSIModel, CSIModelWithZoneMasking

__all__ = [
    'CSIModel',
    'CSIModelWithZoneMasking'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 