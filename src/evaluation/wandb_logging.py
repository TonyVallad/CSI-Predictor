"""
W&B Logging for Evaluation Results

This module provides functions to log evaluation results to Weights & Biases.
"""

import os

# Set environment variables BEFORE importing wandb
# This prevents wandb from creating folders in the project root
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLE_ARTIFACT'] = 'true'
os.environ['WANDB_REQUIRE_SERVICE'] = 'false'

# Now import wandb after setting environment variables
import wandb
from typing import Dict, Any
from src.utils.logging import logger
from src.config import Config

def log_evaluation_results(evaluation_results: Dict[str, Any], config: Config, run_name: str = None) -> None:
    """
    Log evaluation results to Weights & Biases.
    
    Args:
        evaluation_results: Dictionary containing evaluation metrics
        config: Configuration object
        run_name: Optional name for the wandb run
    """
    try:
        # Set wandb directory environment variable globally
        # This prevents wandb from creating a .wandb folder in the current directory
        os.environ['WANDB_DIR'] = config.wandb_dir
        
        # Initialize wandb if not already running
        if wandb.run is None:
            # To avoid nested folders, specify the parent directory
            wandb_parent_dir = os.path.dirname(config.wandb_dir)
            wandb.init(
                project="csi-predictor-eval",
                name=run_name,
                dir=wandb_parent_dir,
                config={
                    "model_arch": config.model_arch,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "optimizer": config.optimizer,
                    "normalization_strategy": config.normalization_strategy,
                }
            )
            logger.info(f"Wandb run initialized with ID: {wandb.run.id}")
            logger.info(f"Wandb directory: {config.wandb_dir}")
            logger.info(f"Wandb parent directory: {wandb_parent_dir}")
        
        # Log evaluation metrics
        wandb.log(evaluation_results)
        logger.info(f"Evaluation results logged to wandb: {list(evaluation_results.keys())}")
        
    except Exception as e:
        logger.error(f"Failed to log evaluation results to wandb: {e}")
        raise

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 