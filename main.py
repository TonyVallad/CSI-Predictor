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
    """Main function to run training and/or evaluation."""
    parser = argparse.ArgumentParser(description="CSI-Predictor: Predict 6-zone CSI scores on chest X-rays")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both",
                        help="Run mode: train, eval, or both")
    parser.add_argument("--config", default="config.ini", help="Path to config.ini file")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    
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
    
    if args.mode in ["train", "both"]:
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