"""
Evaluation CLI for CSI-Predictor.

This module contains evaluation CLI functionality extracted from the original main.py file.
"""

import argparse
from loguru import logger

from src.config import cfg
from src.evaluation.evaluator import evaluate_model

def evaluate_cli(args):
    """
    Handle evaluation CLI commands.
    
    Args:
        args: Parsed command line arguments
    """
    logger.info("Starting evaluation...")
    evaluate_model(cfg)
    logger.info("Evaluation completed.")

def create_evaluate_parser():
    """
    Create evaluation-specific argument parser.
    
    Returns:
        ArgumentParser for evaluation commands
    """
    parser = argparse.ArgumentParser(description="CSI-Predictor Evaluation")
    parser.add_argument("--config", help="Path to config.ini file (if not provided, will use INI_DIR from .env)")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--model-path", help="Path to trained model file")
    parser.add_argument("--output-dir", help="Output directory for evaluation results")
    
    return parser

if __name__ == "__main__":
    parser = create_evaluate_parser()
    args = parser.parse_args()
    evaluate_cli(args)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 