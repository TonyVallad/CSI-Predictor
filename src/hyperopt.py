"""
Hyperparameter Optimization for CSI-Predictor using Optuna.

This module implements comprehensive hyperparameter optimization using Bayesian optimization
with Optuna framework. It includes smart sampling, pruning, and integration with existing
training pipeline and WandB logging.

Usage:
    python -m src.hyperopt --study-name csi_optimization --n-trials 100
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

from .config import Config, get_config
from .data import create_data_loaders
from .models import build_model
from .train import WeightedCSILoss, train_epoch, validate_epoch
from .utils import EarlyStopping, MetricsTracker, logger, seed_everything


# Global cache for data loaders to avoid re-caching images for every trial
_GLOBAL_DATA_CACHE = {}


class OptunaPruningCallback:
    """
    Callback for Optuna pruning integration during training.
    
    This callback allows trials to be pruned early if they're not performing well,
    saving computational resources for more promising hyperparameter combinations.
    """
    
    def __init__(self, trial: optuna.trial.Trial, monitor: str = 'val_f1_macro'):
        """
        Initialize pruning callback.
        
        Args:
            trial: Optuna trial object
            monitor: Metric to monitor for pruning decisions
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
        config.data_dir,
        config.csv_dir,
        config.labels_csv,
        config.load_data_to_memory
    )
    
    # Check if we have cached datasets
    if cache_key not in _GLOBAL_DATA_CACHE:
        logger.info("Creating and caching datasets (this will only happen once)...")
        
        # Create data loaders with a standard batch size first
        temp_config = Config(
            # Copy all values but use standard batch size
            device=config.device,
            load_data_to_memory=config.load_data_to_memory,
            data_source=config.data_source,
            data_dir=config.data_dir,
            models_dir=config.models_dir,
            csv_dir=config.csv_dir,
            ini_dir=config.ini_dir,
            graph_dir=config.graph_dir,
            labels_csv=config.labels_csv,
            labels_csv_separator=config.labels_csv_separator,
            model_arch=config.model_arch,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            batch_size=32,  # Use standard batch size for caching
            patience=config.patience,
            n_epochs=config.n_epochs,
            _env_vars=config._env_vars.copy(),
            _ini_vars=config._ini_vars.copy(),
            _missing_keys=config._missing_keys.copy()
        )
        
        train_loader, val_loader, test_loader = create_data_loaders(temp_config)
        
        # Cache the underlying datasets (not the loaders)
        _GLOBAL_DATA_CACHE[cache_key] = {
            'train_dataset': train_loader.dataset,
            'val_dataset': val_loader.dataset, 
            'test_dataset': test_loader.dataset
        }
        
        logger.info("Datasets cached successfully!")
    
    # Get cached datasets
    cached_data = _GLOBAL_DATA_CACHE[cache_key]
    
    # Create new data loaders with the current batch size
    train_loader = DataLoader(
        cached_data['train_dataset'],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 to avoid multiprocessing issues
    )
    
    val_loader = DataLoader(
        cached_data['val_dataset'],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        cached_data['test_dataset'],
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader


def create_optuna_config(trial: optuna.trial.Trial, base_config: Config) -> Config:
    """
    Create configuration with Optuna-suggested hyperparameters.
    
    Args:
        trial: Optuna trial for hyperparameter suggestions
        base_config: Base configuration to modify
        
    Returns:
        Modified configuration with suggested hyperparameters
    """
    # Model architecture optimization - using only available architectures
    model_arch = trial.suggest_categorical(
        'model_arch', 
        ['ResNet50', 'CheXNet', 'Custom_1']  # Only use actually available ones
    )
    
    # Optimizer hyperparameters
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    
    # Training hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Loss function hyperparameters
    unknown_weight = trial.suggest_float('unknown_weight', 0.1, 1.0)
    
    # Early stopping patience
    patience = trial.suggest_int('patience', 5, 20)
    
    # Optimizer-specific parameters
    weight_decay = 0.0
    momentum = 0.0
    
    if optimizer == 'adamw':
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    elif optimizer == 'sgd':
        momentum = trial.suggest_float('momentum', 0.5, 0.99)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Create modified config (we'll use a temporary approach)
    # Store hyperparameters in a way that can be accessed during training
    hyperparams = {
        'model_arch': model_arch,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'patience': patience,
        'unknown_weight': unknown_weight,
        'weight_decay': weight_decay,
        'momentum': momentum,
    }
    
    # Create a new config with updated values
    # Since Config is frozen, we need to create a new instance
    updated_config = Config(
        # Copy existing values
        device=base_config.device,
        load_data_to_memory=base_config.load_data_to_memory,
        data_source=base_config.data_source,
        data_dir=base_config.data_dir,
        models_dir=base_config.models_dir,
        csv_dir=base_config.csv_dir,
        ini_dir=base_config.ini_dir,
        graph_dir=base_config.graph_dir,
        labels_csv=base_config.labels_csv,
        labels_csv_separator=base_config.labels_csv_separator,
        
        # Updated hyperparameters
        model_arch=model_arch,
        optimizer=optimizer,
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        
        # Keep other training params from base config
        n_epochs=base_config.n_epochs,
        
        # Store hyperparams in internal fields
        _env_vars=base_config._env_vars.copy(),
        _ini_vars={**base_config._ini_vars, **hyperparams},
        _missing_keys=base_config._missing_keys.copy()
    )
    
    return updated_config


def objective(trial: optuna.trial.Trial, base_config: Config, max_epochs: int = 50) -> float:
    """
    Objective function for Optuna optimization.
    
    This function trains a model with trial-suggested hyperparameters and returns
    the validation F1 score for optimization.
    
    Args:
        trial: Optuna trial object
        base_config: Base configuration
        max_epochs: Maximum number of epochs to train
        
    Returns:
        Validation F1 macro score (to maximize)
    """
    try:
        # Set random seeds for reproducibility
        seed_everything(42)
        
        # Create config with suggested hyperparameters
        config = create_optuna_config(trial, base_config)
        
        # Setup device
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Get cached data loaders (this avoids re-caching images)
        train_loader, val_loader, _ = get_cached_data_loaders(config)
        
        # Build model with suggested architecture
        model = build_model(config)
        
        # Create optimizer with suggested parameters
        optimizer_name = config.optimizer.lower()
        optimizer_params = {'lr': config.learning_rate}
        
        # Add optimizer-specific parameters
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), **optimizer_params)
        elif optimizer_name == "adamw":
            weight_decay = config._ini_vars.get('weight_decay', 0.01)
            optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, **optimizer_params)
        elif optimizer_name == "sgd":
            momentum = config._ini_vars.get('momentum', 0.9)
            weight_decay = config._ini_vars.get('weight_decay', 0.0)
            optimizer = optim.SGD(model.parameters(), momentum=momentum, weight_decay=weight_decay, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Create loss function with suggested unknown weight
        unknown_weight = config._ini_vars.get('unknown_weight', 0.3)
        criterion = WeightedCSILoss(unknown_weight=unknown_weight)
        criterion = criterion.to(device)
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, verbose=False
        )
        
        # Early stopping - fix the parameter usage
        early_stopping = EarlyStopping(patience=config.patience, min_delta=0.001)
        
        # Initialize pruning callback
        pruning_callback = OptunaPruningCallback(trial, monitor='val_f1_macro')
        
        # Training loop
        best_val_f1 = 0.0
        
        for epoch in range(1, max_epochs + 1):
            # Train
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            
            # Validate
            val_metrics = validate_epoch(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step(val_metrics["f1_macro"])
            
            # Check for pruning
            if pruning_callback.should_prune(epoch, val_metrics):
                raise optuna.exceptions.TrialPruned()
            
            # Track best validation F1
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
            
            # Early stopping check - use validation loss (lower is better)
            val_loss = val_metrics.get("loss", float('inf'))
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Log trial results
        logger.info(f"Trial {trial.number} completed with best val F1: {best_val_f1:.4f}")
        
        return best_val_f1
    
    except optuna.exceptions.TrialPruned:
        # Handle pruned trials gracefully
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        # Return a poor score for failed trials
        return 0.0


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
        storage_url: Database URL for distributed optimization (optional)
        sampler_name: Sampler algorithm ('tpe', 'random', 'cmaes')
        pruner_name: Pruner algorithm ('median', 'successive_halving', 'none')
        direction: Optimization direction ('maximize' or 'minimize')
        
    Returns:
        Configured Optuna study
    """
    # Configure sampler
    if sampler_name == 'tpe':
        sampler = TPESampler(seed=42, n_startup_trials=10)
    elif sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler(seed=42)
    elif sampler_name == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=42)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")
    
    # Configure pruner
    if pruner_name == 'median':
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif pruner_name == 'successive_halving':
        pruner = SuccessiveHalvingPruner()
    elif pruner_name == 'none':
        pruner = optuna.pruners.NopPruner()
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
    
    logger.info(f"Created study '{study_name}' with {sampler_name} sampler and {pruner_name} pruner")
    return study


def save_best_hyperparameters(study: optuna.Study, output_path: str) -> None:
    """
    Save best hyperparameters to file.
    
    Args:
        study: Completed Optuna study
        output_path: Path to save hyperparameters
    """
    best_params = study.best_params
    best_value = study.best_value
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save hyperparameters
    import json
    with open(output_path, 'w') as f:
        json.dump({
            'best_value': best_value,
            'best_params': best_params,
            'n_trials': len(study.trials),
            'study_name': study.study_name,
            'optimization_time': str(datetime.now())
        }, f, indent=2)
    
    logger.info(f"Best hyperparameters saved to {output_path}")
    logger.info(f"Best validation F1: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")


def clear_data_cache():
    """Clear the global data cache."""
    global _GLOBAL_DATA_CACHE
    _GLOBAL_DATA_CACHE.clear()
    logger.info("Data cache cleared")


def optimize_hyperparameters(
    study_name: str = "csi_optimization",
    n_trials: int = 100,
    max_epochs: int = 50,
    config_path: str = "config.ini",
    sampler: str = 'tpe',
    pruner: str = 'median',
    storage_url: Optional[str] = None,
    wandb_project: Optional[str] = None
) -> optuna.Study:
    """
    Main hyperparameter optimization function.
    
    Args:
        study_name: Name of the optimization study
        n_trials: Number of trials to run
        max_epochs: Maximum epochs per trial
        config_path: Path to configuration file
        sampler: Sampler algorithm to use
        pruner: Pruner algorithm to use
        storage_url: Database URL for distributed optimization
        wandb_project: WandB project name for logging
        
    Returns:
        Completed Optuna study
    """
    # Load base configuration
    base_config = get_config(ini_path=config_path)
    
    # Clear any existing cache
    clear_data_cache()
    
    # Create study
    study = create_study(
        study_name=study_name,
        storage_url=storage_url,
        sampler_name=sampler,
        pruner_name=pruner,
        direction='maximize'
    )
    
    # Setup WandB callback if project specified
    wandb_callback = None
    if wandb_project:
        wandb_callback = WeightsAndBiasesCallback(
            metric_name="val_f1_macro",
            wandb_kwargs={
                "project": wandb_project,
                "name": f"optuna_{study_name}",
                "tags": ["hyperparameter_optimization", "csi_predictor"]
            }
        )
    
    # Define objective function with base config
    def trial_objective(trial):
        return objective(trial, base_config, max_epochs)
    
    # Run optimization
    logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
    logger.info(f"Study name: {study_name}")
    logger.info(f"Max epochs per trial: {max_epochs}")
    logger.info(f"Sampler: {sampler}, Pruner: {pruner}")
    logger.info(f"Available model architectures: ResNet50, CheXNet, Custom_1")
    
    # Optimize
    callbacks = [wandb_callback] if wandb_callback else None
    study.optimize(trial_objective, n_trials=n_trials, callbacks=callbacks)
    
    # Save results
    output_dir = Path(base_config.models_dir) / "hyperopt"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_params_path = output_dir / f"{study_name}_best_params.json"
    save_best_hyperparameters(study, str(best_params_path))
    
    # Print optimization results
    print(f"\n{'='*80}")
    print(f"🎯 HYPERPARAMETER OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Best validation F1: {study.best_value:.4f}")
    print(f"Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study


def main():
    """Main function for CLI entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for CSI-Predictor")
    parser.add_argument("--study-name", default="csi_optimization", help="Name of the optimization study")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of trials to run")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs per trial")
    parser.add_argument("--config", default="config.ini", help="Path to config.ini file")
    parser.add_argument("--sampler", default="tpe", choices=['tpe', 'random', 'cmaes'], help="Sampler algorithm")
    parser.add_argument("--pruner", default="median", choices=['median', 'successive_halving', 'none'], help="Pruner algorithm")
    parser.add_argument("--storage", help="Database URL for distributed optimization")
    parser.add_argument("--wandb-project", help="WandB project name for logging")
    
    args = parser.parse_args()
    
    # Run optimization
    study = optimize_hyperparameters(
        study_name=args.study_name,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        config_path=args.config,
        sampler=args.sampler,
        pruner=args.pruner,
        storage_url=args.storage,
        wandb_project=args.wandb_project
    )
    
    # Optionally create visualizations
    try:
        import optuna.visualization as vis
        import matplotlib.pyplot as plt
        
        output_dir = Path("models/hyperopt")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create optimization history plot
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / f"{args.study_name}_optimization_history.html"))
        
        # Create parameter importance plot
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / f"{args.study_name}_param_importances.html"))
        
        # Create parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / f"{args.study_name}_parallel_coordinate.html"))
        
        logger.info(f"Visualization plots saved to {output_dir}")
        
    except ImportError:
        logger.warning("Visualization dependencies not available. Install plotly for plots.")
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")


if __name__ == "__main__":
    main() 