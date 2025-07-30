"""
Models package for CSI-Predictor.

This package contains model architectures, backbones, and factory functions.
"""

from .factory import build_model, build_zone_focus_model, build_zone_masking_model, get_model_info
from .complete import CSIModel, CSIModelWithZoneMasking

__all__ = [
    'build_model',
    'build_zone_focus_model', 
    'build_zone_masking_model',
    'get_model_info',
    'CSIModel',
    'CSIModelWithZoneMasking'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 