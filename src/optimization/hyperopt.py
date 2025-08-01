"""
Hyperparameter Optimization for CSI-Predictor using Optuna.

This module contains hyperopt functionality extracted from the original src/hyperopt.py file.
"""

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import optuna
from optuna.integration import WeightsAndBiasesCallback
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import TPESampler
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

class OptunaPruningCallback:
    """
    Callback for Optuna pruning integration during training.
    
    This callback allows trials to be pruned early if they're not performing well,
    saving computational resources for more promising hyperparameter combinations.
    """
    
    def __init__(self, trial: optuna.trial.Trial, monitor: str = 'val_f1_weighted'):
        """
        Initialize pruning callback.
        
        Args:
            trial: Optuna trial object
            monitor: Metric to monitor for pruning decisions (default: val_f1_weighted for imbalanced data)
        """
        self.trial = trial
        self.monitor = monitor
    
    def should_prune(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        Check if trial should be pruned based on current performance.
        
        Args:
            epoch: Current epoch number
            metrics: Current validation metrics
            
        Returns:
            True if trial should be pruned
        """
        if self.monitor in metrics:
            # Report intermediate value to Optuna
            self.trial.report(metrics[self.monitor], epoch)
            
            # Check if trial should be pruned
            if self.trial.should_prune():
                logger.info(f"Trial {self.trial.number} pruned at epoch {epoch}")
                return True
        
        return False

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
        logger.info("Creating new data loaders for hyperopt (not cached)")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        _GLOBAL_DATA_CACHE[cache_key] = (train_loader, val_loader, test_loader)
    else:
        logger.info("Using cached data loaders for hyperopt")
    
    return _GLOBAL_DATA_CACHE[cache_key]

def create_optuna_config(trial: optuna.trial.Trial, base_config: Config) -> Config:
    """
    Create configuration with hyperparameters suggested by Optuna trial.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration object
        
    Returns:
        Configuration with suggested hyperparameters
    """
    # Define search spaces for hyperparameters
    suggested_params = {
        # Learning rate (log-uniform distribution)
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        
        # Batch size (categorical for memory efficiency)
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        
        # Optimizer
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
        
        # Weight decay
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        
        # Dropout rate
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.7),
        
        # Model architecture
        'model_arch': trial.suggest_categorical('model_arch', ['resnet50', 'densenet121', 'custom_cnn']),
        
        # Zone focus method
        'zone_focus_method': trial.suggest_categorical('zone_focus_method', ['masking', 'spatial_reduction']),
        
        # Attention strength (if using attention masking)
        'attention_strength': trial.suggest_float('attention_strength', 0.3, 0.9),
    }
    
    # Create new config with suggested parameters
    config_dict = {
        'learning_rate': suggested_params['learning_rate'],
        'batch_size': suggested_params['batch_size'],
        'optimizer': suggested_params['optimizer'],
        'weight_decay': suggested_params['weight_decay'],
        'dropout_rate': suggested_params['dropout_rate'],
        'model_arch': suggested_params['model_arch'],
        'zone_focus_method': suggested_params['zone_focus_method'],
        'attention_strength': suggested_params['attention_strength'],
    }
    
    # Create new config object with suggested parameters
    new_config = Config(
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
        
        # Training Hyperparameters
        n_epochs=base_config.n_epochs,
        patience=base_config.patience,
        
        # Model Configuration
        use_official_processor=base_config.use_official_processor,
        use_segmentation_masking=base_config.use_segmentation_masking,
        masking_strategy=base_config.masking_strategy,
        
        # Image Format Configuration
        image_format=base_config.image_format,
        image_extension=base_config.image_extension,
        
        # Normalization Strategy Configuration
        normalization_strategy=base_config.normalization_strategy,
        custom_mean=base_config.custom_mean,
        custom_std=base_config.custom_std,
        
        # Use suggested parameters
        learning_rate=config_dict['learning_rate'],
        batch_size=config_dict['batch_size'],
        optimizer=config_dict['optimizer'],
        weight_decay=config_dict['weight_decay'],
        dropout_rate=config_dict['dropout_rate'],
        model_arch=config_dict['model_arch'],
        zone_focus_method=config_dict['zone_focus_method'],
        attention_strength=config_dict['attention_strength'],
    )
    
    return new_config

def objective(trial: optuna.trial.Trial, base_config: Config, max_epochs: int = 50) -> float:
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration object
        max_epochs: Maximum number of epochs per trial
        
    Returns:
        Best validation F1 score achieved
    """
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Create configuration with suggested hyperparameters
    config = create_optuna_config(trial, base_config)
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Create callbacks
    early_stopping = EarlyStopping(patience=config.patience, min_delta=0.001)
    metrics_tracker = MetricsTracker()
    pruning_callback = OptunaPruningCallback(trial, monitor='val_f1_weighted')
    
    # Training loop
    best_f1 = 0.0
    
    for epoch in range(max_epochs):
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['val_f1_weighted'])
        
        # Check for pruning
        if pruning_callback.should_prune(epoch, val_metrics):
            raise optuna.TrialPruned()
        
        # Update best F1 score
        if val_metrics['val_f1_weighted'] > best_f1:
            best_f1 = val_metrics['val_f1_weighted']
        
        # Early stopping
        if early_stopping(val_metrics['val_loss'], model):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_f1

def create_study(
    study_name: str,
    storage_url: Optional[str] = None,
    sampler_name: str = 'tpe',
    pruner_name: str = 'median',
    direction: str = 'maximize'
) -> optuna.Study:
    """
    Create Optuna study with specified configuration.
    
    Args:
        study_name: Name of the study
        storage_url: Storage URL for study persistence
        sampler_name: Sampler name ('tpe', 'random', 'cmaes')
        pruner_name: Pruner name ('median', 'successive_halving')
        direction: Optimization direction ('maximize' or 'minimize')
        
    Returns:
        Optuna study object
    """
    # Create sampler
    if sampler_name == 'tpe':
        sampler = TPESampler(seed=42)
    elif sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif sampler_name == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    
    # Create pruner
    if pruner_name == 'median':
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif pruner_name == 'successive_halving':
        pruner = SuccessiveHalvingPruner()
    else:
        raise ValueError(f"Unknown pruner: {pruner_name}")
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True
    )
    
    return study

def save_best_hyperparameters(study: optuna.Study, output_path: str) -> None:
    """
    Save best hyperparameters to file.
    
    Args:
        study: Optuna study object
        output_path: Path to save hyperparameters
    """
    best_params = study.best_params
    best_value = study.best_value
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(f"# Best hyperparameters from Optuna study\n")
        f.write(f"# Best value: {best_value}\n")
        f.write(f"# Study name: {study.study_name}\n\n")
        
        for param, value in best_params.items():
            f.write(f"{param} = {value}\n")
    
    logger.info(f"Best hyperparameters saved to {output_path}")

def clear_data_cache():
    """Clear the global data cache."""
    global _GLOBAL_DATA_CACHE
    _GLOBAL_DATA_CACHE.clear()
    logger.info("Data cache cleared")

def get_search_space_info() -> Dict[str, Any]:
    """
    Get information about the hyperparameter search space.
    
    Returns:
        Dictionary with search space information
    """
    return {
        'learning_rate': {'type': 'float', 'range': [1e-5, 1e-2], 'log': True},
        'batch_size': {'type': 'categorical', 'values': [8, 16, 32, 64]},
        'optimizer': {'type': 'categorical', 'values': ['adam', 'adamw', 'sgd']},
        'weight_decay': {'type': 'float', 'range': [1e-5, 1e-2], 'log': True},
        'dropout_rate': {'type': 'float', 'range': [0.1, 0.7]},
        'model_arch': {'type': 'categorical', 'values': ['resnet50', 'densenet121', 'custom_cnn']},
        'zone_focus_method': {'type': 'categorical', 'values': ['masking', 'spatial_reduction']},
        'attention_strength': {'type': 'float', 'range': [0.3, 0.9]},
    }

def optimize_hyperparameters(
    study_name: str,
    n_trials: int = 100,
    max_epochs: int = 50,
    config_path: Optional[str] = None,
    sampler: str = 'tpe',
    pruner: str = 'median',
    wandb_project: Optional[str] = None
) -> optuna.Study:
    """
    Main function to run hyperparameter optimization.
    
    Args:
        study_name: Name of the study
        n_trials: Number of trials to run
        max_epochs: Maximum epochs per trial
        config_path: Path to configuration file
        sampler: Sampler name ('tpe', 'random', 'cmaes')
        pruner: Pruner name ('median', 'successive_halving')
        wandb_project: WandB project name for logging
        
    Returns:
        Optuna study object
    """
    # Load base configuration
    if config_path:
        from src.config import get_config
        base_config = get_config(ini_path=config_path)
    else:
        from src.config import cfg
        base_config = cfg
    
    # Create study
    study = create_study(
        study_name=study_name,
        sampler_name=sampler,
        pruner_name=pruner,
        direction='maximize'
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config, max_epochs),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Save best hyperparameters
    output_path = f"models/hyperopt/{study_name}_best_params.json"
    save_best_hyperparameters(study, output_path)
    
    logger.info(f"Optimization completed. Best value: {study.best_value}")
    logger.info(f"Best hyperparameters saved to: {output_path}")
    
    return study

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 