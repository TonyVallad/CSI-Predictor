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

def train_sweep_run_enhanced(config: Config, wandb_config: Dict[str, Any]) -> None:
    """
    Enhanced training function for W&B sweep runs with proper initialization and logging.
    
    Args:
        config: Configuration object
        wandb_config: W&B sweep configuration
    """
    import wandb
    import os
    
    # Set wandb environment variables for stability
    os.environ['WANDB_SILENT'] = 'true'
    os.environ['WANDB_DISABLE_ARTIFACT'] = 'true'
    os.environ['WANDB_REQUIRE_SERVICE'] = 'false'
    
    # Set wandb directory environment variable globally
    # This prevents wandb from creating a .wandb folder in the current directory
    os.environ['WANDB_DIR'] = config.wandb_dir
    
    logger.info("Starting enhanced wandb sweep training...")
    
    # Initialize wandb if not already initialized
    if wandb.run is None:
        logger.info("Initializing wandb run for sweep...")
        try:
            wandb.init(dir=config.wandb_dir)
            logger.info(f"Wandb run initialized with ID: {wandb.run.id}")
            logger.info(f"Wandb directory: {config.wandb_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize wandb: {e}")
            raise
    
    # Update configuration with sweep hyperparameters
    config_updates = {}
    for key, value in wandb_config.items():
        if hasattr(config, key):
            setattr(config, key, value)
            config_updates[key] = value
            logger.info(f"Updated config.{key} = {value}")
    
    logger.info(f"Configuration updated with sweep parameters: {config_updates}")
    
    # Run training with enhanced logging
    try:
        from src.training.trainer import train_model
        train_model(config)
        logger.info("Enhanced sweep training completed successfully!")
    except Exception as e:
        logger.error(f"Enhanced sweep training failed: {e}")
        # Log error to wandb
        if wandb.run is not None:
            try:
                wandb.log({'error': str(e)})
            except:
                pass
        raise

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
    Run W&B sweep agent with enhanced error handling.
    
    Args:
        sweep_id: W&B sweep ID
        config: Configuration object
    """
    def train_function():
        try:
            logger.info("Starting sweep run...")
            # Use the enhanced training function
            train_sweep_run_enhanced(config, wandb.config)
            logger.info("Sweep run completed successfully!")
        except Exception as e:
            logger.error(f"Sweep run failed: {e}")
            # Log error to wandb
            if wandb.run is not None:
                try:
                    wandb.log({'error': str(e)})
                except:
                    pass
            raise
    
    logger.info(f"Starting W&B sweep agent for sweep ID: {sweep_id}")
    try:
        wandb.agent(sweep_id, train_function, count=None)  # Run until sweep is complete
        logger.info("Sweep agent completed successfully!")
    except Exception as e:
        logger.error(f"Sweep agent failed: {e}")
        raise

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