"""
W&B Sweeps Implementation for CSI-Predictor.

This module implements hyperparameter optimization using W&B Sweeps instead of Optuna,
providing native WandB integration with automatic visualization and parallelization.

Usage:
    # Initialize a sweep
    python -m src.wandb_sweep --mode init --sweep-name "csi_sweep"
    
    # Run sweep agent
    python -m src.wandb_sweep --mode agent --sweep-id "sweep_id_from_init"
    
    # All-in-one: create and run
    python -m src.wandb_sweep --mode run --sweep-name "csi_sweep" --n-agents 1
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

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


# Global cache for data loaders to avoid re-caching images for every sweep run
_GLOBAL_DATA_CACHE = {}


def get_cached_data_loaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get cached data loaders to avoid re-caching images for every sweep run.
    
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


def get_sweep_config(method: str = "bayes", metric_name: str = "val_f1_macro") -> Dict[str, Any]:
    """
    Generate sweep configuration for W&B.
    
    Args:
        method: Optimization method ('bayes', 'grid', 'random')
        metric_name: Metric to optimize
        
    Returns:
        W&B sweep configuration dictionary
    """
    return {
        'method': method,
        'metric': {
            'goal': 'maximize',
            'name': metric_name
        },
        'parameters': {
            'model_arch': {
                'values': ['ResNet50', 'CheXNet', 'Custom_1', 'RadDINO']
            },
            'optimizer': {
                'values': ['adam', 'adamw', 'sgd']
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-1
            },
            'batch_size': {
                'values': [16, 32, 64, 128]
            },
            'unknown_weight': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'patience': {
                'values': list(range(5, 21))  # 5 to 20
            }
        }
    }


def train_sweep_run(base_config: Config, max_epochs: int = 50):
    """
    Training function for a single sweep run.
    
    This function is called by the W&B sweep agent for each hyperparameter combination.
    
    Args:
        base_config: Base configuration
        max_epochs: Maximum number of epochs to train
    """
    # Initialize W&B run
    with wandb.init() as run:
        # Get hyperparameters from W&B config
        config_dict = dict(wandb.config)
        
        logger.info(f"Starting sweep run {run.name} with config: {config_dict}")
        
        # Set random seeds for reproducibility
        seed_everything(42)
        
        # Create updated config with sweep hyperparameters
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
            
            # Updated hyperparameters from sweep
            model_arch=config_dict.get('model_arch', base_config.model_arch),
            optimizer=config_dict.get('optimizer', base_config.optimizer),
            learning_rate=config_dict.get('learning_rate', base_config.learning_rate),
            batch_size=config_dict.get('batch_size', base_config.batch_size),
            patience=config_dict.get('patience', base_config.patience),
            
            # Keep other training params from base config
            n_epochs=max_epochs,
            
            # Store hyperparams in internal fields
            _env_vars=base_config._env_vars.copy(),
            _ini_vars={**base_config._ini_vars, **config_dict},
            _missing_keys=base_config._missing_keys.copy()
        )
        
        # Setup device
        device = torch.device(updated_config.device if torch.cuda.is_available() else "cpu")
        
        # Get cached data loaders (this avoids re-caching images)
        train_loader, val_loader, _ = get_cached_data_loaders(updated_config)
        
        # Build model with sweep architecture
        model = build_model(updated_config)
        
        # Create optimizer with sweep parameters
        optimizer_name = updated_config.optimizer.lower()
        optimizer_params = {'lr': updated_config.learning_rate}
        
        # Add optimizer-specific parameters
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), **optimizer_params)
        elif optimizer_name == "adamw":
            weight_decay = config_dict.get('weight_decay', 0.01)
            optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, **optimizer_params)
        elif optimizer_name == "sgd":
            momentum = config_dict.get('momentum', 0.9)
            weight_decay = config_dict.get('weight_decay', 0.0)
            optimizer = optim.SGD(model.parameters(), momentum=momentum, weight_decay=weight_decay, **optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Create loss function with sweep unknown weight
        unknown_weight = config_dict.get('unknown_weight', 0.3)
        criterion = WeightedCSILoss(unknown_weight=unknown_weight)
        criterion = criterion.to(device)
        
        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, verbose=False
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=updated_config.patience, min_delta=0.001)
        
        # Training loop
        best_val_f1 = 0.0
        
        for epoch in range(1, max_epochs + 1):
            # Train
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            
            # Validate
            val_metrics = validate_epoch(model, val_loader, criterion, device)
            
            # Update scheduler
            scheduler.step(val_metrics["f1_macro"])
            
            # Log metrics to W&B (native integration - no step conflicts!)
            wandb.log({
                'epoch': epoch,
                'train_loss': train_metrics.get("loss", 0),
                'train_f1_macro': train_metrics.get("f1_macro", 0),
                'val_loss': val_metrics.get("loss", 0),
                'val_f1_macro': val_metrics.get("f1_macro", 0),
                'learning_rate': optimizer.param_groups[0]['lr'],
            })
            
            # Track best validation F1
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                # Log best metrics
                wandb.log({
                    'best_val_f1_macro': best_val_f1,
                    'best_epoch': epoch,
                })
            
            # Early stopping check
            val_loss = val_metrics.get("loss", float('inf'))
            if early_stopping(val_loss, model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                wandb.log({'early_stopped': True, 'early_stop_epoch': epoch})
                break
        
        # Log final results
        wandb.log({
            'final_val_f1_macro': best_val_f1,
            'total_epochs': epoch,
            'completed': True
        })
        
        logger.info(f"Sweep run {run.name} completed with best val F1: {best_val_f1:.4f}")


def initialize_sweep(
    project: str,
    sweep_name: str,
    config_path: str = "config.ini",
    entity: Optional[str] = None
) -> str:
    """
    Initialize a W&B sweep.
    
    Args:
        project: W&B project name
        sweep_name: Name for the sweep
        config_path: Path to configuration file
        entity: W&B entity (username/team)
        
    Returns:
        Sweep ID
    """
    # Load base configuration
    base_config = get_config(ini_path=config_path)
    
    # Clear any existing cache
    global _GLOBAL_DATA_CACHE
    _GLOBAL_DATA_CACHE.clear()
    
    # Create sweep configuration
    sweep_config = get_sweep_config(method="bayes", metric_name="val_f1_macro")
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity
    )
    
    logger.info(f"Initialized W&B sweep: {sweep_id}")
    logger.info(f"Project: {project}")
    logger.info(f"Entity: {entity}")
    logger.info(f"Sweep URL: https://wandb.ai/{entity or 'your-username'}/{project}/sweeps/{sweep_id}")
    
    # Save sweep info for later reference
    output_dir = Path(base_config.models_dir) / "wandb_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_info = {
        'sweep_id': sweep_id,
        'project': project,
        'entity': entity,
        'sweep_name': sweep_name,
        'config': sweep_config,
        'created_at': str(wandb.util.generate_id())
    }
    
    sweep_file = output_dir / f"{sweep_name}_sweep_info.json"
    with open(sweep_file, 'w') as f:
        json.dump(sweep_info, f, indent=2)
    
    logger.info(f"Sweep info saved to: {sweep_file}")
    
    return sweep_id


def run_sweep_agent(
    sweep_id: str,
    project: str,
    config_path: str = "config.ini",
    max_epochs: int = 50,
    count: Optional[int] = None,
    entity: Optional[str] = None
):
    """
    Run a W&B sweep agent.
    
    Args:
        sweep_id: W&B sweep ID
        project: W&B project name
        config_path: Path to configuration file
        max_epochs: Maximum epochs per run
        count: Number of runs to execute (None = infinite)
        entity: W&B entity (username/team)
    """
    # Load base configuration
    base_config = get_config(ini_path=config_path)
    
    logger.info(f"Starting W&B sweep agent")
    logger.info(f"Sweep ID: {sweep_id}")
    logger.info(f"Project: {project}")
    logger.info(f"Max epochs per run: {max_epochs}")
    logger.info(f"Count: {count or 'infinite'}")
    
    # Define the training function for this agent
    def train_function():
        train_sweep_run(base_config, max_epochs)
    
    # Run the sweep agent
    wandb.agent(
        sweep_id=sweep_id,
        function=train_function,
        count=count,
        project=project,
        entity=entity
    )
    
    logger.info("W&B sweep agent completed")


def create_and_run_sweep(
    project: str,
    sweep_name: str,
    n_agents: int = 1,
    max_epochs: int = 50,
    count_per_agent: Optional[int] = None,
    config_path: str = "config.ini",
    entity: Optional[str] = None
) -> str:
    """
    Create a sweep and run agents (all-in-one function).
    
    Args:
        project: W&B project name
        sweep_name: Name for the sweep
        n_agents: Number of agents to run in parallel
        max_epochs: Maximum epochs per run
        count_per_agent: Number of runs per agent
        config_path: Path to configuration file
        entity: W&B entity (username/team)
        
    Returns:
        Sweep ID
    """
    # Initialize sweep
    sweep_id = initialize_sweep(project, sweep_name, config_path, entity)
    
    # Run agents (for now, run them sequentially - can be parallelized later)
    for agent_id in range(n_agents):
        logger.info(f"Starting agent {agent_id + 1}/{n_agents}")
        run_sweep_agent(
            sweep_id=sweep_id,
            project=project,
            config_path=config_path,
            max_epochs=max_epochs,
            count=count_per_agent,
            entity=entity
        )
    
    return sweep_id


def main():
    """Main function for CLI entry point."""
    parser = argparse.ArgumentParser(description="W&B Sweeps for CSI-Predictor")
    parser.add_argument("--mode", choices=["init", "agent", "run"], required=True,
                        help="Mode: init (create sweep), agent (run agent), run (create and run)")
    parser.add_argument("--project", default="csi-sweeps", help="W&B project name")
    parser.add_argument("--sweep-name", default="csi_sweep", help="Name for the sweep")
    parser.add_argument("--sweep-id", help="W&B sweep ID (for agent mode)")
    parser.add_argument("--entity", help="W&B entity (username/team)")
    parser.add_argument("--config", default="config.ini", help="Path to config.ini file")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs per run")
    parser.add_argument("--n-agents", type=int, default=1, help="Number of agents (for run mode)")
    parser.add_argument("--count", type=int, help="Number of runs per agent")
    
    args = parser.parse_args()
    
    if args.mode == "init":
        sweep_id = initialize_sweep(
            project=args.project,
            sweep_name=args.sweep_name,
            config_path=args.config,
            entity=args.entity
        )
        print(f"\n{'='*80}")
        print(f"🚀 W&B SWEEP INITIALIZED")
        print(f"{'='*80}")
        print(f"Sweep ID: {sweep_id}")
        print(f"Next step: Run an agent with:")
        print(f"  python -m src.wandb_sweep --mode agent --sweep-id {sweep_id}")
        print(f"Or visit: https://wandb.ai/{args.entity or 'your-username'}/{args.project}/sweeps/{sweep_id}")
        
    elif args.mode == "agent":
        if not args.sweep_id:
            parser.error("--sweep-id is required for agent mode")
        
        run_sweep_agent(
            sweep_id=args.sweep_id,
            project=args.project,
            config_path=args.config,
            max_epochs=args.max_epochs,
            count=args.count,
            entity=args.entity
        )
        
    elif args.mode == "run":
        sweep_id = create_and_run_sweep(
            project=args.project,
            sweep_name=args.sweep_name,
            n_agents=args.n_agents,
            max_epochs=args.max_epochs,
            count_per_agent=args.count,
            config_path=args.config,
            entity=args.entity
        )
        print(f"\n{'='*80}")
        print(f"🎯 W&B SWEEP COMPLETED")
        print(f"{'='*80}")
        print(f"Sweep ID: {sweep_id}")
        print(f"View results: https://wandb.ai/{args.entity or 'your-username'}/{args.project}/sweeps/{sweep_id}")


if __name__ == "__main__":
    main() 