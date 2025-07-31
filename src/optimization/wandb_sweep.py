"""
Weights & Biases sweep integration for CSI-Predictor.

This module contains W&B sweep functionality extracted from the original src/wandb_sweep.py file.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from ..config import Config, get_config
from ..data.dataloader import create_data_loaders
from ..models.factory import build_model
from ..training.loss import WeightedCSILoss
from ..training.trainer import train_epoch, validate_epoch
from ..training.callbacks import EarlyStopping, MetricsTracker
from ..utils.logging import logger
from ..utils.seed import seed_everything

# Global cache for data loaders to avoid re-caching images for every trial
_GLOBAL_DATA_CACHE = {}

def get_cached_data_loaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get cached data loaders to avoid re-caching images for every trial.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    global _GLOBAL_DATA_CACHE
    
    # Create cache key based on data configuration (but not batch size)
    cache_key = (
        config.nifti_dir,
        config.csv_dir,
        config.labels_csv,
        config.load_data_to_memory
    )
    
    if cache_key not in _GLOBAL_DATA_CACHE:
        logger.info("Creating new data loaders for W&B sweep (not cached)")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        _GLOBAL_DATA_CACHE[cache_key] = (train_loader, val_loader, test_loader)
    else:
        logger.info("Using cached data loaders for W&B sweep")
    
    return _GLOBAL_DATA_CACHE[cache_key]

def get_sweep_config() -> Dict[str, Any]:
    """
    Get W&B sweep configuration.
    
    Returns:
        Dictionary with sweep configuration
    """
    return {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_f1_weighted',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-5,
                'max': 1e-2,
                'distribution': 'log_uniform'
            },
            'batch_size': {
                'values': [8, 16, 32, 64]
            },
            'optimizer': {
                'values': ['adam', 'adamw', 'sgd']
            },
            'weight_decay': {
                'min': 1e-5,
                'max': 1e-2,
                'distribution': 'log_uniform'
            },
            'dropout_rate': {
                'min': 0.1,
                'max': 0.7
            },
            'model_arch': {
                'values': ['resnet50', 'densenet121', 'custom_cnn']
            },
            'zone_focus_method': {
                'values': ['masking', 'spatial_reduction']
            },
            'attention_strength': {
                'min': 0.3,
                'max': 0.9
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        }
    }

def train_sweep_run(config: Config, wandb_config: Dict[str, Any]) -> None:
    """
    Train a single W&B sweep run.
    
    Args:
        config: Base configuration object
        wandb_config: W&B configuration with hyperparameters
    """
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Update config with W&B hyperparameters
    config = Config(
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
        
        # Training Hyperparameters
        n_epochs=config.n_epochs,
        patience=config.patience,
        
        # Model Configuration
        use_official_processor=config.use_official_processor,
        use_segmentation_masking=config.use_segmentation_masking,
        masking_strategy=config.masking_strategy,
        masks_path=config.masks_path,
        image_format=config.image_format,
        image_extension=config.image_extension,
        normalization_strategy=config.normalization_strategy,
        custom_mean=config.custom_mean,
        custom_std=config.custom_std,
        
        # Use W&B hyperparameters
        learning_rate=wandb_config['learning_rate'],
        batch_size=wandb_config['batch_size'],
        optimizer=wandb_config['optimizer'],
        weight_decay=wandb_config['weight_decay'],
        dropout_rate=wandb_config['dropout_rate'],
        model_arch=wandb_config['model_arch'],
        zone_focus_method=wandb_config['zone_focus_method'],
        attention_strength=wandb_config['attention_strength'],
    )
    
    # Get cached data loaders
    train_loader, val_loader, _ = get_cached_data_loaders(config)
    
    # Build model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.to(device)
    
    # Create loss function
    criterion = WeightedCSILoss()
    
    # Create optimizer
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Create callbacks
    early_stopping = EarlyStopping(patience=config.patience, min_delta=0.001)
    metrics_tracker = MetricsTracker()
    
    # Training loop
    for epoch in range(config.n_epochs):
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['val_f1_weighted'])
        
        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['train_loss'],
            'train_f1_weighted': train_metrics['train_f1_weighted'],
            'val_loss': val_metrics['val_loss'],
            'val_f1_weighted': val_metrics['val_f1_weighted'],
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping
        if early_stopping(val_metrics['val_loss'], model):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

def initialize_sweep(project_name: str, sweep_config: Dict[str, Any]) -> str:
    """
    Initialize W&B sweep.
    
    Args:
        project_name: W&B project name
        sweep_config: Sweep configuration
        
    Returns:
        Sweep ID
    """
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    logger.info(f"Initialized W&B sweep with ID: {sweep_id}")
    return sweep_id

def run_sweep_agent(sweep_id: str, config: Config) -> None:
    """
    Run W&B sweep agent.
    
    Args:
        sweep_id: W&B sweep ID
        config: Configuration object
    """
    def train_function():
        with wandb.init() as run:
            # Get hyperparameters from W&B
            wandb_config = wandb.config
            
            # Train the model
            train_sweep_run(config, wandb_config)
    
    # Run the agent
    wandb.agent(sweep_id, train_function, count=None)  # Run until sweep is complete

def create_and_run_sweep(
    project_name: str = "csi-sweep",
    n_runs: int = 100,
    config: Optional[Config] = None
) -> str:
    """
    Create and run W&B sweep.
    
    Args:
        project_name: W&B project name
        n_runs: Number of runs in the sweep
        config: Configuration object (uses default if None)
        
    Returns:
        Sweep ID
    """
    if config is None:
        config = get_config()
    
    # Get sweep configuration
    sweep_config = get_sweep_config()
    sweep_config['name'] = f"csi_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize sweep
    sweep_id = initialize_sweep(project_name, sweep_config)
    
    # Run sweep agent
    run_sweep_agent(sweep_id, config)
    
    return sweep_id

def clear_data_cache():
    """Clear the global data cache."""
    global _GLOBAL_DATA_CACHE
    _GLOBAL_DATA_CACHE.clear()
    logger.info("Data cache cleared")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 