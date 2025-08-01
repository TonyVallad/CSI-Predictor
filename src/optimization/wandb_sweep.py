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

def get_sweep_config(model_arch: str = 'resnet50') -> Dict[str, Any]:
    """
    Get W&B sweep configuration for a specific model architecture.
    
    Args:
        model_arch: Model architecture ('resnet50', 'chexnet', 'raddino')
        
    Returns:
        Dictionary with sweep configuration
    """
    # Model-specific configurations
    model_configs = {
        'resnet50': {
            'batch_size': 128,
            'learning_rate': {'min': 0.0001, 'max': 0.01},
            'use_official_processor': False
        },
        'chexnet': {
            'batch_size': 64,
            'learning_rate': {'min': 0.00005, 'max': 0.005},
            'use_official_processor': False
        },
        'raddino': {
            'batch_size': 8,
            'learning_rate': {'min': 0.00001, 'max': 0.001},
            'use_official_processor': True
        }
    }
    
    config = model_configs.get(model_arch, model_configs['resnet50'])
    
    return {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_f1_weighted',
            'goal': 'maximize'
        },
        'parameters': {
            # Fixed model parameters
            'model_arch': {
                'value': model_arch
            },
            'use_official_processor': {
                'value': config['use_official_processor']
            },
            'batch_size': {
                'value': config['batch_size']
            },
            
            # Sweep parameters
            'optimizer': {
                'values': ['adam', 'adamw', 'sgd']
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': config['learning_rate']['min'],
                'max': config['learning_rate']['max']
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 0.000001,
                'max': 0.001
            },
            'dropout_rate': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.7
            },
            'momentum': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 0.99
            },
            'normalization_strategy': {
                'values': ['imagenet', 'medical']
            },
            'scheduler_type': {
                'values': ['ReduceLROnPlateau', 'CosineAnnealingLR']
            },
            'unknown_weight': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'patience': {
                'value': 15
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,
            'eta': 3
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
        
        # Model Configuration
        use_official_processor=config.use_official_processor,
        use_segmentation_masking=config.use_segmentation_masking,
        masking_strategy=config.masking_strategy,
        masks_path=config.masks_path,
        image_format=config.image_format,
        image_extension=config.image_extension,
        custom_mean=config.custom_mean,
        custom_std=config.custom_std,
        
        # Use W&B hyperparameters
        learning_rate=wandb_config['learning_rate'],
        batch_size=wandb_config['batch_size'],
        optimizer=wandb_config['optimizer'],
        weight_decay=wandb_config['weight_decay'],
        dropout_rate=wandb_config['dropout_rate'],
        model_arch=wandb_config['model_arch'],
        normalization_strategy=wandb_config['normalization_strategy'],
        patience=wandb_config['patience'],
    )
    
    # Get cached data loaders
    train_loader, val_loader, _ = get_cached_data_loaders(config)
    
    # Load CSV data for metrics computation
    import pandas as pd
    csv_path = Path(config.csv_dir) / config.labels_csv
    csv_data = pd.read_csv(csv_path, sep=config.labels_csv_separator) if csv_path.exists() else None
    
    # Build model
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.to(device)
    
    # Create loss function with unknown weight from sweep
    unknown_weight = wandb_config.get('unknown_weight', 0.5)
    criterion = WeightedCSILoss(unknown_weight=unknown_weight)
    
    # Create optimizer
    if config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    # Create scheduler based on sweep configuration
    scheduler_type = wandb_config.get('scheduler_type', 'ReduceLROnPlateau')
    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_epochs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    # Create callbacks
    early_stopping = EarlyStopping(patience=config.patience, min_delta=0.001)
    metrics_tracker = MetricsTracker()
    
    # Training loop
    for epoch in range(config.n_epochs):
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, csv_data, config)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device, csv_data, config)
        
        # Update scheduler based on type
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(val_metrics['f1_weighted_overall'])
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler.step()
        
        # Log to W&B
        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_f1_weighted': train_metrics['f1_weighted_overall'],
            'val_loss': val_metrics['loss'],
            'val_f1_weighted': val_metrics['f1_weighted_overall'],
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Debug logging
        logger.info(f"Epoch {epoch}: val_f1_weighted = {val_metrics['f1_weighted_overall']}")
        
        # Early stopping based on validation F1 score
        if early_stopping(val_metrics['f1_weighted_overall'], model):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Log final metric for W&B sweep optimization
    final_val_f1 = val_metrics['f1_weighted_overall']
    wandb.log({'val_f1_weighted': final_val_f1})
    logger.info(f"Final val_f1_weighted: {final_val_f1}")

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
        with wandb.init(dir=config.wandb_dir) as run:
            # Get hyperparameters from W&B
            wandb_config = wandb.config
            
            # Train the model
            train_sweep_run(config, wandb_config)
    
    # Run the agent
    wandb.agent(sweep_id, train_function, count=None)  # Run until sweep is complete

def create_and_run_sweep(
    project_name: str = "csi-sweep",
    model_arch: str = "resnet50",
    n_runs: int = 100,
    config: Optional[Config] = None
) -> str:
    """
    Create and run W&B sweep for a specific model architecture.
    
    Args:
        project_name: W&B project name
        model_arch: Model architecture ('resnet50', 'chexnet', 'raddino')
        n_runs: Number of runs in the sweep
        config: Configuration object (uses default if None)
        
    Returns:
        Sweep ID
    """
    if config is None:
        config = get_config()
    
    # Get sweep configuration for the specific model
    sweep_config = get_sweep_config(model_arch)
    sweep_config['name'] = f"CSI-Predictor {model_arch.upper()} Hyperparameter Optimization"
    
    # Initialize sweep
    sweep_id = initialize_sweep(project_name, sweep_config)
    
    # Run sweep agent
    run_sweep_agent(sweep_id, config)
    
    return sweep_id

def initialize_model_sweep(
    project_name: str = "csi-predictor",
    model_arch: str = "resnet50"
) -> str:
    """
    Initialize a W&B sweep for a specific model architecture.
    
    Args:
        project_name: W&B project name
        model_arch: Model architecture ('resnet50', 'chexnet', 'raddino')
        
    Returns:
        Sweep ID
    """
    # Get sweep configuration for the specific model
    sweep_config = get_sweep_config(model_arch)
    sweep_config['name'] = f"CSI-Predictor {model_arch.upper()} Hyperparameter Optimization"
    
    # Initialize sweep
    sweep_id = initialize_sweep(project_name, sweep_config)
    
    logger.info(f"Initialized sweep for {model_arch}: {sweep_id}")
    logger.info(f"Sweep URL: https://wandb.ai/{project_name}/sweeps/{sweep_id}")
    
    return sweep_id

def clear_data_cache():
    """Clear the global data cache."""
    global _GLOBAL_DATA_CACHE
    _GLOBAL_DATA_CACHE.clear()
    logger.info("Data cache cleared")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 