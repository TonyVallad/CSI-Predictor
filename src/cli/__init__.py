"""
CLI package for CSI-Predictor.

This package contains all CLI-related functionality including:
- Main CLI entry point
- Training CLI
- Evaluation CLI
- Optimization CLI
"""

from .main import main
from .train import train_cli, create_train_parser
from .evaluate import evaluate_cli, create_evaluate_parser
from .optimize import optimize_cli, create_optimize_parser

__all__ = [
    'main',
    'train_cli',
    'create_train_parser',
    'evaluate_cli',
    'create_evaluate_parser',
    'optimize_cli',
    'create_optimize_parser'
]

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 