"""
Training CLI for CSI-Predictor.

This module contains training CLI functionality extracted from the original main.py file.
"""

import argparse
from loguru import logger

from src.config import cfg, copy_config_on_training_start
from src.training.trainer import train_model

def train_cli(args, optimized=False):
    """
    Handle training CLI commands.
    
    Args:
        args: Parsed command line arguments
        optimized: Whether to use optimized hyperparameters
    """
    if optimized:
        if not args.hyperparams:
            logger.error("--hyperparams argument required for train-optimized mode")
            logger.info("Run hyperparameter optimization first:")
            logger.info(f"  python main.py --mode hyperopt --study-name {args.study_name} --n-trials {args.n_trials}")
            return
        
        logger.info("Training with optimized hyperparameters...")
        from src.training.train_optimized import train_with_optimized_hyperparameters
        
        train_with_optimized_hyperparameters(
            hyperparams_path=args.hyperparams,
            config_path=args.config,
            full_epochs=True
        )
        logger.info("Training with optimized hyperparameters completed.")
    else:
        logger.info("Starting training...")
        # Copy configuration with timestamp for reproducibility
        copy_config_on_training_start()
        train_model(cfg)
        logger.info("Training completed.")

def create_train_parser():
    """
    Create training-specific argument parser.
    
    Returns:
        ArgumentParser for training commands
    """
    parser = argparse.ArgumentParser(description="CSI-Predictor Training")
    parser.add_argument("--config", help="Path to config.ini file (if not provided, will use INI_DIR from .env)")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    parser.add_argument("--hyperparams", help="Path to JSON file with best hyperparameters (for optimized training)")
    parser.add_argument("--optimized", action="store_true", help="Use optimized hyperparameters")
    
    return parser

if __name__ == "__main__":
    parser = create_train_parser()
    args = parser.parse_args()
    train_cli(args, optimized=args.optimized)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 