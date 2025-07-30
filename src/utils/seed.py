"""
Random seed management for CSI-Predictor.

This module contains random seed functionality extracted from the original src/utils.py file.
"""

import torch
import numpy as np
import random
from loguru import logger

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_everything(seed: int = 42) -> None:
    """
    Set random seed for all libraries for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set random seed to {seed} for all libraries")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 