"""
Train CSI-Predictor model using optimized hyperparameters from Optuna.

This script loads the best hyperparameters found by Optuna optimization
and trains the final model with those settings for the full number of epochs.

Usage:
    python -m src.training.train_optimized --hyperparams models/hyperopt/csi_optimization_best_params.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim

from src.config import Config, get_config
from src.data.dataloader import create_data_loaders
from src.models.factory import build_model
from .loss import WeightedCSILoss
from .trainer import train_model
from src.utils.logging import logger
from src.utils.seed import seed_everything


def load_best_hyperparameters(hyperparams_path: str) -> Dict[str, Any]:
    """
    Load best hyperparameters from Optuna optimization results.
    
    Args:
        hyperparams_path: Path to JSON file with best hyperparameters
        
    Returns:
        Dictionary of best hyperparameters
    """
    with open(hyperparams_path, 'r') as f:
        results = json.load(f)
    
    best_params = results['best_params']
    best_value = results['best_value']
    
    logger.info(f"Loaded optimized hyperparameters from {hyperparams_path}")
    logger.info(f"Best validation F1 during optimization: {best_value:.4f}")
    logger.info(f"Best hyperparameters: {best_params}")
    
    return best_params


def create_optimized_config(base_config: Config, best_params: Dict[str, Any]) -> Config:
    """
    Create configuration with optimized hyperparameters.
    
    Args:
        base_config: Base configuration
        best_params: Best hyperparameters from Optuna
        
    Returns:
        Updated configuration with optimized hyperparameters
    """
    # Create optimized config with new hyperparameters
    optimized_config = Config(
        # Environment and Device Settings
        device=base_config.device,
        load_data_to_memory=base_config.load_data_to_memory,
        
        # Data Paths
        data_source=base_config.data_source,
        data_dir=base_config.data_dir,
        nifti_dir=base_config.nifti_dir,
        models_dir=base_config.models_dir,
        csv_dir=base_config.csv_dir,
        ini_dir=base_config.ini_dir,
        png_dir=base_config.png_dir,
        graph_dir=base_config.graph_dir,
        debug_dir=base_config.debug_dir,
        masks_dir=base_config.masks_dir,
        logs_dir=base_config.logs_dir,
        runs_dir=base_config.runs_dir,
        evaluation_dir=base_config.evaluation_dir,
        wandb_dir=base_config.wandb_dir,
        
        # Labels configuration
        labels_csv=base_config.labels_csv,
        labels_csv_separator=base_config.labels_csv_separator,
        
        # Data Filtering
        excluded_file_ids=base_config.excluded_file_ids,
        
        # Training Hyperparameters (optimized)
        batch_size=best_params.get('batch_size', base_config.batch_size),
        n_epochs=base_config.n_epochs,
        patience=base_config.patience,
        learning_rate=best_params.get('learning_rate', base_config.learning_rate),
        optimizer=best_params.get('optimizer', base_config.optimizer),
        dropout_rate=best_params.get('dropout_rate', base_config.dropout_rate),
        weight_decay=best_params.get('weight_decay', base_config.weight_decay),
        
        # Model Configuration
        model_arch=base_config.model_arch,
        use_official_processor=base_config.use_official_processor,
        zone_focus_method=base_config.zone_focus_method,
        
        # Zone Masking Configuration
        use_segmentation_masking=base_config.use_segmentation_masking,
        masking_strategy=base_config.masking_strategy,
        attention_strength=base_config.attention_strength,
        
        # Image Format Configuration
        image_format=base_config.image_format,
        image_extension=base_config.image_extension,
        
        # Normalization Strategy Configuration
        normalization_strategy=base_config.normalization_strategy,
        custom_mean=base_config.custom_mean,
        custom_std=base_config.custom_std
    )
    
    return optimized_config


def train_with_optimized_hyperparameters(
    hyperparams_path: str,
    config_path: str = "config.ini",
    full_epochs: bool = True
) -> None:
    """
    Train model using optimized hyperparameters.
    
    Args:
        hyperparams_path: Path to best hyperparameters JSON file
        config_path: Path to base configuration file
        full_epochs: Whether to train for full epochs (vs. reduced for optimization)
    """
    # Set random seeds for reproducibility
    seed_everything(42)
    
    # Load base configuration and best hyperparameters
    base_config = get_config(ini_path=config_path)
    best_params = load_best_hyperparameters(hyperparams_path)
    
    # Create optimized configuration
    config = create_optimized_config(base_config, best_params)
    
    # If training for full epochs, potentially increase from optimization setting
    if full_epochs and config.n_epochs < 100:
        logger.info(f"Increasing epochs from {config.n_epochs} to 100 for final training")
        # We need to modify the config, but it's frozen. Let's create a new one.
        optimized_config = Config(
            # Environment and Device Settings
            device=config.device,
            load_data_to_memory=config.load_data_to_memory,
            
            # Data Paths
            data_source=config.data_source,
            data_dir=config.data_dir,
            nifti_dir=config.nifti_dir,
            models_dir=config.models_dir,
            csv_dir=config.csv_dir,
            ini_dir=config.ini_dir,
            png_dir=config.png_dir,
            graph_dir=config.graph_dir,
            debug_dir=config.debug_dir,
            masks_dir=config.masks_dir,
            logs_dir=config.logs_dir,
            runs_dir=config.runs_dir,
            evaluation_dir=config.evaluation_dir,
            wandb_dir=config.wandb_dir,
            
            # Labels configuration
            labels_csv=config.labels_csv,
            labels_csv_separator=config.labels_csv_separator,
            
            # Data Filtering
            excluded_file_ids=config.excluded_file_ids,
            
            # Training Hyperparameters (optimized)
            batch_size=best_params.get('batch_size', config.batch_size),
            n_epochs=config.n_epochs,
            patience=config.patience,
            learning_rate=best_params.get('learning_rate', config.learning_rate),
            optimizer=best_params.get('optimizer', config.optimizer),
            dropout_rate=best_params.get('dropout_rate', config.dropout_rate),
            weight_decay=best_params.get('weight_decay', config.weight_decay),
            
            # Model Configuration
            model_arch=config.model_arch,
            use_official_processor=config.use_official_processor,
            zone_focus_method=config.zone_focus_method,
            
            # Zone Masking Configuration
            use_segmentation_masking=config.use_segmentation_masking,
            masking_strategy=config.masking_strategy,
            attention_strength=config.attention_strength,
            
            # Image Format Configuration
            image_format=config.image_format,
            image_extension=config.image_extension,
            
            # Normalization Strategy Configuration
            normalization_strategy=config.normalization_strategy,
            custom_mean=config.custom_mean,
            custom_std=config.custom_std
        )
    
    # Log the optimized configuration
    logger.info("Training with optimized hyperparameters:")
    logger.info(f"  Model architecture: {config.model_arch}")
    logger.info(f"  Optimizer: {config.optimizer}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Unknown weight: {best_params.get('unknown_weight', 0.3)}")
    logger.info(f"  Weight decay: {best_params.get('weight_decay', 0.0)}")
    logger.info(f"  Momentum: {best_params.get('momentum', 0.0)}")
    logger.info(f"  Patience: {config.patience}")
    logger.info(f"  Epochs: {config.n_epochs}")
    
    # Train the model using the existing train_model function
    # The train_model function will automatically use the configuration values
    train_model(config)
    
    # Save hyperparameters alongside the trained model
    models_dir = Path(config.models_dir)
    hyperparams_save_path = models_dir / "final_model_hyperparams.json"
    
    with open(hyperparams_save_path, 'w') as f:
        json.dump({
            'source_optimization': hyperparams_path,
            'used_hyperparameters': best_params,
            'training_config': {
                'model_arch': config.model_arch,
                'optimizer': config.optimizer,
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'n_epochs': config.n_epochs,
                'patience': config.patience
            }
        }, f, indent=2)
    
    logger.info(f"Final model hyperparameters saved to {hyperparams_save_path}")
    logger.info("🎉 Training with optimized hyperparameters completed!")


def main():
    """Main function for CLI entry point."""
    parser = argparse.ArgumentParser(description="Train CSI-Predictor with optimized hyperparameters")
    parser.add_argument(
        "--hyperparams", 
        required=True,
        help="Path to JSON file with best hyperparameters from Optuna"
    )
    parser.add_argument(
        "--config", 
        default="config.ini", 
        help="Path to base config.ini file"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Use the same number of epochs as optimization (don't increase for final training)"
    )
    
    args = parser.parse_args()
    
    # Check if hyperparameters file exists
    if not Path(args.hyperparams).exists():
        logger.error(f"Hyperparameters file not found: {args.hyperparams}")
        logger.info("Run hyperparameter optimization first:")
        logger.info("  python -m src.hyperopt --study-name csi_optimization --n-trials 50")
        return
    
    # Train with optimized hyperparameters
    train_with_optimized_hyperparameters(
        hyperparams_path=args.hyperparams,
        config_path=args.config,
        full_epochs=not args.quick
    )


if __name__ == "__main__":
    main() 