"""
Training logic for CSI-Predictor.
Handles model training, validation, and checkpointing with classification loss.
"""

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import wandb

from .config import cfg, get_config, copy_config_on_training_start
from .data import create_data_loaders
from .models import build_model
from .utils import EarlyStopping, MetricsTracker, logger, seed_everything, make_run_name, log_config
from .metrics import compute_pytorch_f1_metrics


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    seed_everything(seed)
    logger.info(f"Set random seeds to {seed}")


class MaskedCrossEntropyLoss(nn.Module):
    """
    Masked Cross-Entropy Loss that ignores positions where ground-truth == 4.
    
    This handles "ungradable" or "unknown" CSI zones by not including them
    in the loss computation.
    """
    
    def __init__(self, ignore_index: int = 4):
        """
        Initialize masked cross-entropy loss.
        
        Args:
            ignore_index: Class index to ignore (default: 4 for ungradable)
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute masked cross-entropy loss.
        
        Args:
            predictions: Model predictions [batch_size, n_zones, n_classes]
            targets: Ground truth labels [batch_size, n_zones]
            
        Returns:
            Scalar loss value
        """
        batch_size, n_zones, n_classes = predictions.shape
        
        # Reshape for cross-entropy: [batch_size * n_zones, n_classes]
        predictions_flat = predictions.view(-1, n_classes)
        targets_flat = targets.view(-1)
        
        # Compute loss per sample
        losses = self.cross_entropy(predictions_flat, targets_flat)
        
        # Create mask for valid (non-ignored) targets
        mask = (targets_flat != self.ignore_index)
        
        # Apply mask and compute mean only over valid samples
        if mask.sum() > 0:
            masked_losses = losses[mask]
            return masked_losses.mean()
        else:
            # If all samples are masked, return zero loss
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)


def compute_f1_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute F1 scores for CSI prediction using PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        
    Returns:
        Dictionary with F1 metrics
    """
    return compute_pytorch_f1_metrics(predictions, targets, ignore_index=4)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        
    Returns:
        Training metrics
    """
    model.train()
    metrics = MetricsTracker()
    
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # [batch_size, n_zones, n_classes]
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        metrics.update("loss", loss.item())
        
        # Store for F1 computation
        all_predictions.append(outputs.detach())
        all_targets.append(targets.detach())
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute F1 metrics
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    f1_metrics = compute_f1_metrics(all_pred_tensor, all_target_tensor)
    
    # Combine metrics
    train_metrics = metrics.get_averages()
    train_metrics.update(f1_metrics)
    
    return train_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        Validation metrics
    """
    model.eval()
    metrics = MetricsTracker()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation"):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Track metrics
            metrics.update("loss", loss.item())
            
            # Store for F1 computation
            all_predictions.append(outputs)
            all_targets.append(targets)
    
    # Compute F1 metrics
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    f1_metrics = compute_f1_metrics(all_pred_tensor, all_target_tensor)
    
    # Combine metrics
    val_metrics = metrics.get_averages()
    val_metrics.update(f1_metrics)
    
    return val_metrics


def train_model(config) -> None:
    """
    Main training function with enhanced logging and utilities.
    
    Args:
        config: Configuration object
    """
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Log configuration
    log_config(config)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, _ = create_data_loaders(config)
    logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
    
    # Build model
    model = build_model(config)
    
    # Create optimizer
    optimizer_name = config.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Create loss function (masked cross-entropy)
    criterion = MaskedCrossEntropyLoss(ignore_index=4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Initialize wandb
    use_wandb = False
    try:
        run_name = f"train_{make_run_name(config)}"
        wandb.init(
            project="csi-predictor",
            name=run_name,
            config={
                "model_arch": config.model_arch,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "optimizer": config.optimizer,
                "n_epochs": config.n_epochs,
                "patience": config.patience,
                "device": config.device,
                "models_dir": config.models_dir,
                "run_name": run_name
            }
        )
        use_wandb = True
        logger.info(f"Initialized Weights & Biases logging with run name: {run_name}")
    except Exception as e:
        logger.warning(f"Could not initialize wandb: {e}")
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    for epoch in range(1, config.n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{config.n_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log metrics
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Log to wandb
        if use_wandb:
            wandb_log = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_f1_macro": train_metrics["f1_macro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "train_f1_overall": train_metrics["f1_overall"],
                "val_f1_overall": val_metrics["f1_overall"]
            }
            
            # Add per-zone F1 scores
            for i in range(6):
                zone_name = f"zone_{i+1}"
                wandb_log[f"train_f1_{zone_name}"] = train_metrics[f"f1_{zone_name}"]
                wandb_log[f"val_f1_{zone_name}"] = val_metrics[f"f1_{zone_name}"]
            
            wandb.log(wandb_log)
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1_macro"]
            
            # Create model name with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_name = f"{config.model_arch} - {timestamp}"
            
            save_path = Path(config.get_model_path(model_name))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'val_f1_macro': val_metrics["f1_macro"],
                'config': config,
                'model_name': model_name,
                'run_name': run_name if use_wandb else None
            }, save_path)
            
            logger.info(f"Saved best model: {model_name}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}")
            
            # Log model artifact to wandb
            if use_wandb:
                artifact = wandb.Artifact(f"model-{run_name}", type="model")
                artifact.add_file(str(save_path))
                wandb.log_artifact(artifact)
        
        # Early stopping check
        if early_stopping(val_metrics["loss"]):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation F1: {best_val_f1:.4f}")
    
    if use_wandb:
        # Log final summary
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_val_f1"] = best_val_f1
        wandb.summary["total_epochs"] = epoch
        wandb.finish()


def main():
    """Main function for CLI entry point."""
    parser = argparse.ArgumentParser(description="Train CSI-Predictor model")
    parser.add_argument("--ini", default="config.ini", help="Path to config.ini file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(ini_path=args.ini)
    
    # Copy configuration for reproducibility
    copy_config_on_training_start()
    
    # Start training
    train_model(config)


if __name__ == "__main__":
    main() 