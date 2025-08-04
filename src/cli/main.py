"""
Main CLI entry point for CSI-Predictor.

This module contains the main CLI functionality extracted from the original main.py file.
"""

import argparse
import os
import sys
from pathlib import Path

# Set environment variables BEFORE importing wandb
# This prevents wandb from creating folders in the project root
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLE_ARTIFACT'] = 'true'
os.environ['WANDB_REQUIRE_SERVICE'] = 'false'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from loguru import logger

from src.config import cfg, copy_config_on_training_start
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from .train import train_cli
from .evaluate import evaluate_cli
from .optimize import optimize_cli

import wandb

def main():
    """Main function to run training, evaluation, and/or hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="CSI-Predictor: Predict 6-zone CSI scores on chest X-rays")
    parser.add_argument("--mode", choices=["train", "eval", "both", "hyperopt", "train-optimized", "sweep", "sweep-agent", "sweep-train"], default="both",
                        help="Run mode: train, eval, both, hyperopt (Optuna), train-optimized, sweep (W&B Sweeps), sweep-agent, or sweep-train")
    parser.add_argument("--config", help="Path to config.ini file (if not provided, will use INI_DIR from .env)")
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
    elif args.mode == "sweep-train":
        # Special mode for wandb sweep training - following best practices
        
        # Set wandb directory environment variable globally
        # This prevents wandb from creating a .wandb folder in the current directory
        os.environ['WANDB_DIR'] = cfg.wandb_dir
        
        logger.info("Starting wandb sweep training...")
        
        # Initialize wandb if not already initialized
        if wandb.run is None:
            logger.info("Initializing wandb run...")
            try:
                # To avoid nested folders, specify the parent directory
                wandb_parent_dir = os.path.dirname(cfg.wandb_dir)
                wandb.init(dir=wandb_parent_dir)
                logger.info(f"Wandb run initialized with ID: {wandb.run.id}")
                logger.info(f"Wandb directory: {cfg.wandb_dir}")
                logger.info(f"Wandb parent directory: {wandb_parent_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
                raise
        
        # Log the wandb config for debugging
        logger.info(f"Wandb config: {dict(wandb.config)}")
        
        # Update configuration with wandb hyperparameters
        config_updates = {}
        if hasattr(wandb.config, 'learning_rate'):
            config_updates['learning_rate'] = wandb.config.learning_rate
        if hasattr(wandb.config, 'batch_size'):
            config_updates['batch_size'] = wandb.config.batch_size
        if hasattr(wandb.config, 'optimizer'):
            config_updates['optimizer'] = wandb.config.optimizer
        if hasattr(wandb.config, 'weight_decay'):
            config_updates['weight_decay'] = wandb.config.weight_decay
        if hasattr(wandb.config, 'dropout_rate'):
            config_updates['dropout_rate'] = wandb.config.dropout_rate
        if hasattr(wandb.config, 'model_arch'):
            config_updates['model_arch'] = wandb.config.model_arch
        if hasattr(wandb.config, 'normalization_strategy'):
            config_updates['normalization_strategy'] = wandb.config.normalization_strategy
        if hasattr(wandb.config, 'patience'):
            config_updates['patience'] = wandb.config.patience
        if hasattr(wandb.config, 'use_official_processor'):
            config_updates['use_official_processor'] = wandb.config.use_official_processor
        
        # Apply configuration updates
        for key, value in config_updates.items():
            setattr(cfg, key, value)
            logger.info(f"Updated config.{key} = {value}")
        
        # Run training with enhanced wandb logging
        try:
            train_model(cfg)
            logger.info("Wandb sweep training completed successfully!")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Log error to wandb
            if wandb.run is not None:
                wandb.log({'error': str(e)})
            raise
    elif args.mode in ["train", "both"]:
        train_cli(args)
    elif args.mode == "eval":
        evaluate_cli(args)

if __name__ == "__main__":
    main()

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 