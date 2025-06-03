"""
Main entry point for CSI-Predictor.
Run the complete training and evaluation pipeline using configuration from .env, config.ini, and config.py.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.config import cfg, copy_config_on_training_start
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    """Main function to run training, evaluation, and/or hyperparameter optimization."""
    parser = argparse.ArgumentParser(description="CSI-Predictor: Predict 6-zone CSI scores on chest X-rays")
    parser.add_argument("--mode", choices=["train", "eval", "both", "hyperopt", "train-optimized"], default="both",
                        help="Run mode: train, eval, both, hyperopt (hyperparameter optimization), or train-optimized")
    parser.add_argument("--config", default="config.ini", help="Path to config.ini file")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    
    # Hyperparameter optimization specific arguments
    parser.add_argument("--study-name", default="csi_optimization", help="Name of the Optuna study")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum epochs per trial during optimization")
    parser.add_argument("--sampler", default="tpe", choices=['tpe', 'random', 'cmaes'], help="Optuna sampler algorithm")
    parser.add_argument("--pruner", default="median", choices=['median', 'successive_halving', 'none'], help="Optuna pruner algorithm")
    parser.add_argument("--wandb-project", default="csi-hyperopt", help="WandB project name for hyperopt logging (default: csi-hyperopt). Set to 'none' to disable WandB logging.")
    
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
    
    if args.mode == "hyperopt":
        logger.info("Starting hyperparameter optimization...")
        from src.hyperopt import optimize_hyperparameters
        
        # Handle WandB project configuration
        wandb_project = args.wandb_project if args.wandb_project.lower() != 'none' else None
        
        if wandb_project:
            logger.info(f"WandB logging enabled - Project: {wandb_project}")
            logger.info("You can monitor progress in real-time at: https://wandb.ai/")
        else:
            logger.info("WandB logging disabled")
        
        study = optimize_hyperparameters(
            study_name=args.study_name,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            config_path=args.config,
            sampler=args.sampler,
            pruner=args.pruner,
            wandb_project=wandb_project
        )
        logger.info("Hyperparameter optimization completed.")
        
        # Print recommended next steps
        print(f"\n{'='*80}")
        print("🎯 NEXT STEPS:")
        print("="*80)
        print("1. Review the optimization results above")
        print("2. Train the final model with optimized hyperparameters:")
        print(f"   python main.py --mode train-optimized --hyperparams models/hyperopt/{args.study_name}_best_params.json")
        print("3. Or use the train_optimized script directly:")
        print(f"   python -m src.train_optimized --hyperparams models/hyperopt/{args.study_name}_best_params.json")
        
    elif args.mode == "train-optimized":
        if not args.hyperparams:
            logger.error("--hyperparams argument required for train-optimized mode")
            logger.info("Run hyperparameter optimization first:")
            logger.info(f"  python main.py --mode hyperopt --study-name {args.study_name} --n-trials {args.n_trials}")
            return
        
        logger.info("Training with optimized hyperparameters...")
        from src.train_optimized import train_with_optimized_hyperparameters
        
        train_with_optimized_hyperparameters(
            hyperparams_path=args.hyperparams,
            config_path=args.config,
            full_epochs=True
        )
        logger.info("Training with optimized hyperparameters completed.")
        
    elif args.mode in ["train", "both"]:
        logger.info("Starting training...")
        # Copy configuration with timestamp for reproducibility
        copy_config_on_training_start()
        train_model(cfg)
        logger.info("Training completed.")
    
    if args.mode in ["eval", "both"]:
        logger.info("Starting evaluation...")
        evaluate_model(cfg)
        logger.info("Evaluation completed.")


if __name__ == "__main__":
    main() 