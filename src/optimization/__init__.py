"""
Optimization package for CSI-Predictor.

This package contains all hyperparameter optimization functionality including:
- Optuna hyperparameter optimization
- W&B sweep integration
"""

from .hyperopt import optimize_hyperparameters
from .wandb_sweep import initialize_sweep, run_sweep_agent

__all__ = [
    'optimize_hyperparameters',
    'initialize_sweep',
    'run_sweep_agent'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 