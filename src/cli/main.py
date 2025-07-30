"""
Main CLI entry point for CSI-Predictor.

This module contains the main CLI functionality extracted from the original main.py file.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from ..config import cfg, copy_config_on_training_start
from ..training.trainer import train_model
from ..evaluation.evaluator import evaluate_model
from .train import train_cli
from .evaluate import evaluate_cli
from .optimize import optimize_cli

def main():
    """Main function to run training, evaluation, and/or hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="CSI-Predictor: Predict 6-zone CSI scores on chest X-rays")
    parser.add_argument("--mode", choices=["train", "eval", "both", "hyperopt", "train-optimized", "sweep", "sweep-agent"], default="both",
                        help="Run mode: train, eval, both, hyperopt (Optuna), train-optimized, sweep (W&B Sweeps), or sweep-agent")
    parser.add_argument("--config", default="config/config.ini", help="Path to config.ini file")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    
    # Hyperparameter optimization specific arguments (Optuna)
    parser.add_argument("--study-name", default="csi_optimization", help="Name of the Optuna study")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum epochs per trial during optimization")
    parser.add_argument("--sampler", default="tpe", choices=['tpe', 'random', 'cmaes'], help="Optuna sampler algorithm")
    parser.add_argument("--pruner", default="median", choices=['median', 'successive_halving', 'none'], help="Optuna pruner algorithm")
    parser.add_argument("--wandb-project", default="csi-hyperopt", help="WandB project name for hyperopt logging (default: csi-hyperopt). Set to 'none' to disable WandB logging.")
    
    # W&B Sweeps specific arguments
    parser.add_argument("--sweep-project", default="csi-sweeps", help="W&B project name for sweeps")
    parser.add_argument("--sweep-name", default="csi_sweep", help="Name for the W&B sweep")
    parser.add_argument("--sweep-id", help="W&B sweep ID (for sweep-agent mode)")
    parser.add_argument("--entity", help="W&B entity (username/team)")
    parser.add_argument("--count", type=int, help="Number of runs for sweep agent")
    
    # Train with optimized hyperparameters
    parser.add_argument("--hyperparams", help="Path to JSON file with best hyperparameters (for train-optimized mode)")
    
    args = parser.parse_args()
    
    # Load environment variables for backward compatibility
    if Path(args.env).exists():
        load_dotenv(args.env)
        logger.info(f"Loaded environment variables from {args.env}")
    else:
        logger.warning(f"Environment file not found: {args.env}")
    
    # Display current configuration
    logger.info("Current Configuration:")
    logger.info(f"  Device: {cfg.device}")
    logger.info(f"  Model Architecture: {cfg.model_arch}")
    logger.info(f"  Batch Size: {cfg.batch_size}")
    logger.info(f"  Learning Rate: {cfg.learning_rate}")
    logger.info(f"  Epochs: {cfg.n_epochs}")
    logger.info(f"  Data Path: {cfg.data_path}")
    logger.info(f"  Models Folder: {cfg.models_folder}")
    logger.info(f"  Model Path: {cfg.get_model_path('best_model')}")
    
    # Route to appropriate CLI module
    if args.mode == "hyperopt":
        optimize_cli(args)
    elif args.mode == "train-optimized":
        train_cli(args, optimized=True)
    elif args.mode == "sweep":
        optimize_cli(args, mode="sweep")
    elif args.mode == "sweep-agent":
        optimize_cli(args, mode="sweep-agent")
    elif args.mode in ["train", "both"]:
        train_cli(args)
    
    if args.mode in ["eval", "both"]:
        evaluate_cli(args)

if __name__ == "__main__":
    main()

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 