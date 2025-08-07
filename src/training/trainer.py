"""
Training Module for CSI-Predictor

This module contains the main training loop and related functions.
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Set environment variables BEFORE importing wandb
# This prevents wandb from creating folders in the project root
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLE_ARTIFACT'] = 'true'
os.environ['WANDB_REQUIRE_SERVICE'] = 'false'

# Now import wandb after setting environment variables
import wandb

from ..config import Config
from ..data.dataloader import create_data_loaders
from ..data.preprocessing import load_csv_data, convert_nans_to_unknown
from ..models.factory import build_model
from ..utils.seed import seed_everything
from ..utils.logging import logger
from ..utils.file_utils import create_dirs, save_training_history
from ..evaluation.visualization.plots import plot_training_curves
from ..utils.checkpoint import save_checkpoint
from ..utils.discord_notifier import send_training_notification
from .loss import WeightedCSILoss
from .metrics import compute_f1_metrics, compute_f1_metrics_with_unknown, compute_precision_recall, compute_csi_average_metrics, compute_ahf_classification_metrics
from .callbacks import EarlyStopping, MetricsTracker
from .optimizer import create_optimizer, create_scheduler

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
    
    # Compute metrics
    # Check if all targets are unknown class (4)
    unique_targets = torch.unique(all_target_tensor)
    if len(unique_targets) == 1 and unique_targets[0] == 4:
        logger.warning("All targets are unknown class (4). Using F1 computation that includes unknown class.")
        f1_metrics = compute_f1_metrics_with_unknown(all_pred_tensor, all_target_tensor)
    else:
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
    
    # Check if all targets are unknown class (4)
    unique_targets = torch.unique(all_target_tensor)
    if len(unique_targets) == 1 and unique_targets[0] == 4:
        logger.warning("All targets are unknown class (4). Using F1 computation that includes unknown class.")
        from .metrics import compute_f1_metrics_with_unknown
        f1_metrics = compute_f1_metrics_with_unknown(all_pred_tensor, all_target_tensor)
    else:
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
    # Check if we're in a wandb sweep context
    import wandb
    is_wandb_run = wandb.run is not None
    
    # Note: Config should already be updated with wandb hyperparameters in sweep_train.py
    # No need to modify config here since it's already mutable
    if is_wandb_run:
        logger.info(f"Wandb config received: {dict(wandb.config)}")
    
    # Create run directory for this training session
    from src.utils.file_utils import create_run_directory
    run_dir = create_run_directory(config, run_type="train")
    
    # Copy configuration to run directory (handled by copy_config_to_run_dir below)
    
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
    model = build_model(config)
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Setup loss function with unknown weight from sweep if available
    unknown_weight = 0.3  # default
    if is_wandb_run and hasattr(wandb.config, 'unknown_weight'):
        unknown_weight = wandb.config.unknown_weight
    
    criterion = WeightedCSILoss(unknown_weight=unknown_weight)
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
        
        # Log to wandb if in sweep context
        if is_wandb_run:
            try:
                # Get the key metric for sweep optimization
                val_f1_weighted = val_metrics.get('f1_weighted_overall', val_metrics.get('f1_macro', 0.0))
                train_f1_weighted = train_metrics.get('f1_weighted_overall', train_metrics.get('f1_macro', 0.0))
                
                # Ensure valid values - handle NaN and Inf
                if isinstance(val_f1_weighted, (int, float)):
                    if math.isnan(val_f1_weighted) or math.isinf(val_f1_weighted):
                        val_f1_weighted = 0.0
                else:
                    val_f1_weighted = 0.0
                    
                if isinstance(train_f1_weighted, (int, float)):
                    if math.isnan(train_f1_weighted) or math.isinf(train_f1_weighted):
                        train_f1_weighted = 0.0
                else:
                    train_f1_weighted = 0.0
                
                # Create log dictionary with all metrics
                log_dict = {
                    'epoch': epoch,
                    'train_loss': float(train_metrics['loss']),
                    'train_f1_weighted': float(train_f1_weighted),
                    'val_loss': float(val_metrics['loss']),
                    'val_f1_weighted': float(val_f1_weighted),  # This is the key metric for the sweep
                    'learning_rate': float(current_lr),
                    'train_f1_macro': float(train_metrics['f1_macro']),
                    'val_f1_macro': float(val_metrics['f1_macro']),
                    'train_accuracy': float(train_metrics['accuracy']),
                    'val_accuracy': float(val_metrics['accuracy']),
                }
                
                # Log to wandb with explicit step
                wandb.log(log_dict, step=epoch)
                
                logger.info(f"Successfully logged to wandb at epoch {epoch}: val_f1_weighted = {val_f1_weighted}")
                logger.info(f"Wandb run ID: {wandb.run.id}")
                
            except Exception as e:
                logger.error(f"Failed to log to wandb at epoch {epoch}: {e}")
                logger.error(f"Log dict: {log_dict}")
                # Try to log just the error
                try:
                    wandb.log({'error': str(e)}, step=epoch)
                except:
                    pass
        else:
            logger.info(f"Not in wandb context - skipping wandb logging for epoch {epoch}")
        
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
        
        # Generate heatmaps for current epoch if enabled
        if config.heatmap_enabled and config.heatmap_generate_per_epoch:
            try:
                heatmaps_dir = run_dir / "heatmaps"
                from src.evaluation.visualization.heatmaps import generate_heatmaps_for_epoch
                generate_heatmaps_for_epoch(model, val_loader, str(heatmaps_dir), config, epoch)
            except Exception as e:
                logger.error(f"Failed to generate heatmaps for epoch {epoch}: {e}")
        
        # Check early stopping
        if early_stopping(val_metrics["loss"], model):
            logger.info(f"Early stopping triggered after {epoch} epochs")
            break
    
    # Log final metric for wandb sweep optimization
    if is_wandb_run:
        try:
            # Use the best validation F1 score as the final metric
            final_val_f1 = float(best_val_f1)
            
            # Ensure it's a valid number
            if math.isnan(final_val_f1) or math.isinf(final_val_f1):
                final_val_f1 = 0.0
                logger.warning("Best val_f1_weighted was NaN/Inf, setting to 0.0")
            
            # Log the final metric - this is what the sweep uses for optimization
            wandb.log({'val_f1_weighted': final_val_f1}, step=config.n_epochs)
            
            logger.info(f"Final val_f1_weighted logged to wandb: {final_val_f1}")
            logger.info(f"Wandb run ID: {wandb.run.id}")
            logger.info("Sweep run completed successfully!")
            
            # Also log a summary
            wandb.log({
                'best_val_f1_weighted': final_val_f1,
                'final_epoch': config.n_epochs,
                'training_completed': True
            }, step=config.n_epochs)
            
        except Exception as e:
            logger.error(f"Failed to log final metric to wandb: {e}")
            # Try to log the error
            try:
                wandb.log({'error': str(e)}, step=config.n_epochs)
            except:
                pass
    else:
        logger.info("Not in wandb context - skipping final metric logging")
    
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
    
    # Save training history to INI_DIR (like config.ini)
    from src.utils.file_utils import save_training_history_to_ini_dir
    
    # Get final AHF confusion matrices from the last validation epoch
    final_train_ahf_cm = None
    final_val_ahf_cm = None
    
    # For now, we'll use the validation AHF confusion matrix as both train and val
    # since we don't track training AHF metrics separately during training
    if 'ahf_confusion_matrix' in val_metrics:
        final_val_ahf_cm = val_metrics['ahf_confusion_matrix']
        final_train_ahf_cm = val_metrics['ahf_confusion_matrix']  # Use same for now
    
    ini_history_path = save_training_history_to_ini_dir(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_precisions, val_precisions,
        train_f1_scores, val_f1_scores,
        config,
        final_train_ahf_cm,
        final_val_ahf_cm
    )
    
    # Copy training history to run directory
    from src.utils.file_utils import copy_training_history_to_run_dir
    copy_training_history_to_run_dir(run_dir, config)
    
    # Copy config.ini to run directory
    from src.utils.file_utils import copy_config_to_run_dir
    copy_config_to_run_dir(run_dir, config)
    
    # Plot training curves to run directory
    training_curves_dir = run_dir / "graphs" / "training_curves"
    plot_training_curves(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_f1_scores, val_f1_scores,
        str(training_curves_dir), "training_run",
        train_precisions, val_precisions
    )
    
    # Generate heatmaps for best model
    if config.heatmap_enabled:
        try:
            heatmaps_dir = run_dir / "heatmaps"
            from src.evaluation.visualization.heatmaps import generate_heatmaps_for_best_model
            
            # Load the best model for heatmap generation
            best_model_path = None
            if best_val_f1 > 0:
                # Find the best F1 model
                for file in os.listdir(config.models_dir):
                    if file.startswith("best_f1_model_epoch_"):
                        best_model_path = os.path.join(config.models_dir, file)
                        break
            
            if best_model_path and os.path.exists(best_model_path):
                # Load best model
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded best model from {best_model_path} for heatmap generation")
            
            generate_heatmaps_for_best_model(model, val_loader, str(heatmaps_dir), config)
            logger.info(f"Heatmaps saved to: {heatmaps_dir}")
            
        except Exception as e:
            logger.error(f"Failed to generate heatmaps: {e}")
    
    logger.info("Training completed successfully!")
    logger.info(f"All outputs saved to run directory: {run_dir}")
    
    return run_dir

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 