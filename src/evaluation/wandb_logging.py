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

def log_to_wandb(
    val_metrics: Dict,
    test_metrics: Dict,
    val_confusion_matrices: Dict,
    test_confusion_matrices: Dict,
    val_reports: Dict,
    test_reports: Dict,
    config,
    model_path: str,
    eval_model_name: str
) -> None:
    """
    Log evaluation results to Weights & Biases.
    
    Args:
        val_metrics: Validation metrics
        test_metrics: Test metrics
        val_confusion_matrices: Validation confusion matrices
        test_confusion_matrices: Test confusion matrices
        val_reports: Validation classification reports
        test_reports: Test classification reports
        config: Configuration object
        model_path: Path to the evaluated model
        eval_model_name: Name of the evaluation run
    """
    try:
        # Set wandb directory environment variable globally
        os.environ['WANDB_DIR'] = config.wandb_dir
        
        # Initialize wandb run
        wandb_parent_dir = os.path.dirname(config.wandb_dir)
        wandb.init(
            project="csi-predictor",
            name=eval_model_name,
            dir=wandb_parent_dir,
            config={
                "model_path": model_path,
                "model_arch": config.model_arch,
                "batch_size": config.batch_size,
                "device": config.device,
                "evaluation_type": "model_evaluation"
            }
        )
        
        # Log overall metrics
        wandb.log({
            "validation/overall_f1_macro": val_metrics.get('f1_macro', 0.0),
            "validation/overall_accuracy": val_metrics.get('accuracy', 0.0),
            "validation/overall_precision_macro": val_metrics.get('precision_macro', 0.0),
            "validation/overall_recall_macro": val_metrics.get('recall_macro', 0.0),
            "test/overall_f1_macro": test_metrics.get('f1_macro', 0.0),
            "test/overall_accuracy": test_metrics.get('accuracy', 0.0),
            "test/overall_precision_macro": test_metrics.get('precision_macro', 0.0),
            "test/overall_recall_macro": test_metrics.get('recall_macro', 0.0),
        })
        
        # Log per-zone metrics
        zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
        
        for zone_name in zone_names:
            if zone_name in val_metrics:
                wandb.log({
                    f"validation/{zone_name}_f1": val_metrics[zone_name].get('f1_macro', 0.0),
                    f"validation/{zone_name}_accuracy": val_metrics[zone_name].get('accuracy', 0.0),
                    f"test/{zone_name}_f1": test_metrics[zone_name].get('f1_macro', 0.0),
                    f"test/{zone_name}_accuracy": test_metrics[zone_name].get('accuracy', 0.0),
                })
        
        # Finish wandb run
        wandb.finish()
        
        logger.info(f"Successfully logged evaluation results to W&B for {eval_model_name}")
        
    except Exception as e:
        logger.warning(f"Could not log to W&B: {e}")
        logger.info("Continuing without W&B logging")

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