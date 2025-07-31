"""
Main training logic for CSI-Predictor.

This module contains the main training functionality extracted from the original src/train.py file.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import wandb
import pandas as pd

from src.config import Config, cfg
from src.data.dataloader import create_data_loaders
from src.data.preprocessing import load_csv_data, convert_nans_to_unknown
from src.models.factory import build_model
from src.utils.seed import seed_everything
from src.utils.logging import logger
from src.utils.file_utils import create_dirs, save_training_history
from src.evaluation.visualization.plots import plot_training_curves
from src.utils.checkpoint import save_checkpoint
from src.utils.discord_notifier import send_training_notification
from .loss import WeightedCSILoss
from .metrics import compute_f1_metrics, compute_precision_recall, compute_csi_average_metrics, compute_ahf_classification_metrics
from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .callbacks import EarlyStopping, MetricsTracker

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    seed_everything(seed)
    logger.info(f"Set random seeds to {seed}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    csv_data: Optional[pd.DataFrame] = None,
    config = None
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
        csv_data: CSV data for metrics computation
        config: Configuration object
        
    Returns:
        Training metrics
    """
    model.train()
    metrics = MetricsTracker()
    
    all_predictions = []
    all_targets = []
    all_file_ids = []
    
    # Check if model supports zone masking
    is_zone_masking_model = hasattr(model, 'ZONE_MAPPING')
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # Handle both old and new data formats
        if len(batch_data) == 3:  # New format: (images, targets, file_ids)
            images, targets, file_ids = batch_data
        else:  # Old format: (images, targets)
            images, targets = batch_data
            file_ids = None
        
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Use zone masking if model supports it and file_ids are available
        if is_zone_masking_model and file_ids is not None:
            outputs = model(images, file_ids)  # [batch_size, n_zones, n_classes]
        else:
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
        if file_ids is not None:
            all_file_ids.extend(file_ids)
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute F1 metrics and accuracy
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    f1_metrics = compute_f1_metrics(all_pred_tensor, all_target_tensor)
    pr_metrics = compute_precision_recall(all_pred_tensor, all_target_tensor)
    
    # Compute accuracy
    pred_classes = torch.argmax(all_pred_tensor, dim=-1)
    valid_mask = all_target_tensor != 4  # Exclude ungradable samples
    accuracy = (pred_classes[valid_mask] == all_target_tensor[valid_mask]).float().mean().item()
    
    # Compute CSI average metrics
    csi_avg_metrics = compute_csi_average_metrics(
        all_pred_tensor, all_target_tensor, 
        file_ids=all_file_ids if all_file_ids else None,
        csv_data=csv_data
    )
    
    # Compute AHF classification metrics
    ahf_metrics = compute_ahf_classification_metrics(
        all_pred_tensor, all_target_tensor, 
        file_ids=all_file_ids if all_file_ids else None,
        csv_data=csv_data
    )
    
    # Combine metrics
    train_metrics = metrics.get_averages()
    train_metrics.update(f1_metrics)
    train_metrics.update(pr_metrics)
    train_metrics.update(csi_avg_metrics)
    train_metrics.update(ahf_metrics)
    train_metrics['accuracy'] = accuracy
    
    return train_metrics


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    csv_data: Optional[pd.DataFrame] = None,
    config = None
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        csv_data: CSV data for metrics computation
        config: Configuration object
        
    Returns:
        Validation metrics
    """
    model.eval()
    metrics = MetricsTracker()
    
    all_predictions = []
    all_targets = []
    all_file_ids = []
    
    # Check if model supports zone masking
    is_zone_masking_model = hasattr(model, 'ZONE_MAPPING')
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # Handle both old and new data formats
            if len(batch_data) == 3:  # New format: (images, targets, file_ids)
                images, targets, file_ids = batch_data
            else:  # Old format: (images, targets)
                images, targets = batch_data
                file_ids = None
            
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            if is_zone_masking_model and file_ids is not None:
                outputs = model(images, file_ids)  # [batch_size, n_zones, n_classes]
            else:
                outputs = model(images)  # [batch_size, n_zones, n_classes]
            
            loss = criterion(outputs, targets)
            
            # Track metrics
            metrics.update("loss", loss.item())
            
            # Store for F1 computation
            all_predictions.append(outputs.detach())
            all_targets.append(targets.detach())
            if file_ids is not None:
                all_file_ids.extend(file_ids)
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute F1 metrics and accuracy
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    f1_metrics = compute_f1_metrics(all_pred_tensor, all_target_tensor)
    pr_metrics = compute_precision_recall(all_pred_tensor, all_target_tensor)
    
    # Compute accuracy
    pred_classes = torch.argmax(all_pred_tensor, dim=-1)
    valid_mask = all_target_tensor != 4  # Exclude ungradable samples
    accuracy = (pred_classes[valid_mask] == all_target_tensor[valid_mask]).float().mean().item()
    
    # Compute CSI average metrics
    csi_avg_metrics = compute_csi_average_metrics(
        all_pred_tensor, all_target_tensor, 
        file_ids=all_file_ids if all_file_ids else None,
        csv_data=csv_data
    )
    
    # Compute AHF classification metrics
    ahf_metrics = compute_ahf_classification_metrics(
        all_pred_tensor, all_target_tensor, 
        file_ids=all_file_ids if all_file_ids else None,
        csv_data=csv_data
    )
    
    # Combine metrics
    val_metrics = metrics.get_averages()
    val_metrics.update(f1_metrics)
    val_metrics.update(pr_metrics)
    val_metrics.update(csi_avg_metrics)
    val_metrics.update(ahf_metrics)
    val_metrics['accuracy'] = accuracy
    
    return val_metrics


def train_model(config: Config) -> Path:
    """
    Main training function.
    
    Args:
        config: Configuration object
        
    Returns:
        Path to the run directory
    """
    # Create run directory for this training session
    from src.utils.file_utils import create_run_directory
    run_dir = create_run_directory(config, run_type="train")
    
    # Copy configuration to run directory
    from src.config import copy_config_on_training_start
    copy_config_on_training_start(run_dir)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    from src.data.dataloader import create_data_loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Load CSV data for file ID tracking
    csv_data = None
    try:
        csv_path = os.path.join(config.csv_dir, config.labels_csv)
        csv_data = load_csv_data(csv_path)
        csv_data = convert_nans_to_unknown(csv_data)
        logger.info(f"Loaded CSV data with {len(csv_data)} samples for CSI average comparison")
    except Exception as e:
        logger.warning(f"Could not load CSV data for CSI average comparison: {e}")
        logger.info("CSI average metrics will be computed without CSV ground truth comparison")
    
    # Create model
    from src.models.factory import build_model
    model = build_model(config)
    model = model.to(device)
    
    # Setup optimizer and scheduler
    from src.training.optimizer import create_optimizer
    from src.training.scheduler import create_scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Setup loss function
    from src.training.loss import WeightedCSILoss
    criterion = WeightedCSILoss(unknown_weight=0.3)  # 30% weight for unknown class
    criterion = criterion.to(device)  # Move criterion to same device as model
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Create output directories
    create_dirs(config.models_dir, config.logs_dir, config.graph_dir, config.debug_dir)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    
    # Track metrics for plotting curves
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    
    for epoch in range(1, config.n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{config.n_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, csv_data, config)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, csv_data, config)
        
        # Update scheduler
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Store metrics for plotting
        train_losses.append(train_metrics['loss'])
        val_losses.append(val_metrics['loss'])
        train_f1_scores.append(train_metrics['f1_macro'])
        val_f1_scores.append(val_metrics['f1_macro'])
        train_accuracies.append(train_metrics['accuracy'])
        val_accuracies.append(val_metrics['accuracy'])
        train_precisions.append(train_metrics['precision_macro'])
        val_precisions.append(val_metrics['precision_macro'])
        
        # Log metrics
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model_path = os.path.join(config.models_dir, f"best_model_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics["loss"], model_path, config=config)
            logger.info(f"Saved best model (loss) to {model_path}")
        
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            model_path = os.path.join(config.models_dir, f"best_f1_model_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, val_metrics["f1_macro"], model_path, config=config)
            logger.info(f"Saved best model (F1) to {model_path}")
        
        # Check early stopping
        if early_stopping(val_metrics["loss"], model):
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    final_model_path = os.path.join(config.models_dir, "final_model.pth")
    save_checkpoint(model, optimizer, config.n_epochs, val_metrics["loss"], final_model_path, config=config)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save training history to run directory
    history_path = run_dir / "training_history" / "training_history.json"
    save_training_history(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_precisions, val_precisions,
        train_f1_scores, val_f1_scores,
        str(history_path)
    )
    
    # Plot training curves to run directory
    training_curves_dir = run_dir / "graphs" / "training_curves"
    plot_training_curves(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_f1_scores, val_f1_scores,
        str(training_curves_dir), "training_run",
        train_precisions, val_precisions
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"All outputs saved to run directory: {run_dir}")
    
    return run_dir

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 