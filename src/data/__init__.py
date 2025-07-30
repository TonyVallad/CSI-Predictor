"""
Data pipeline package for CSI-Predictor.

This package contains all data-related functionality including:
- Dataset classes
- Data loaders
- Image transformations
- Data preprocessing
- Data splitting utilities
"""

from .preprocessing import get_normalization_parameters, CSI_COLUMNS, CSI_UNKNOWN_CLASS
from .dataloader import create_data_loaders
from .dataset import CSIDataset
from .transforms import get_transforms
from .splitting import split_data_stratified

__all__ = [
    'get_normalization_parameters',
    'CSI_COLUMNS',
    'CSI_UNKNOWN_CLASS',
    'create_data_loaders',
    'CSIDataset',
    'get_transforms',
    'split_data_stratified'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 