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
from .utils import EarlyStopping, MetricsTracker, logger, seed_everything, make_run_name, make_model_name, log_config, plot_training_curves
from .metrics import compute_pytorch_f1_metrics, compute_precision_recall_metrics, compute_enhanced_f1_metrics


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    seed_everything(seed)
    logger.info(f"Set random seeds to {seed}")


class WeightedCSILoss(nn.Module):
    """
    Weighted Cross-Entropy Loss that reduces importance of unknown class
    but still learns to predict it.
    
    This treats "ungradable" or "unknown" CSI zones (class 4) as a valid 
    prediction target rather than ignoring it, but gives it reduced weight
    to emphasize learning the clear CSI classifications (0-3).
    """
    
    def __init__(self, unknown_weight: float = 0.3):
        """
        Initialize weighted cross-entropy loss.
        
        Args:
            unknown_weight: Weight for unknown class (default: 0.3)
                          - 1.0 = equal importance with other classes
                          - 0.5 = half importance  
                          - 0.1 = very low importance
        """
        super().__init__()
        self.unknown_weight = unknown_weight
        
        # Weights: [Normal, Mild, Moderate, Severe, Unknown]
        # Classes 0-3 get full weight, class 4 gets reduced weight
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, unknown_weight])
        
        # Register weights as buffer so they move to device automatically
        self.register_buffer('class_weights', weights)
        
        # Initialize CrossEntropyLoss without weights initially
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        
        logger.info(f"Initialized WeightedCSILoss with unknown_weight={unknown_weight}")
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.
        
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
        
        # Compute weighted cross-entropy using F.cross_entropy with weights
        import torch.nn.functional as F
        return F.cross_entropy(predictions_flat, targets_flat, weight=self.class_weights)


def compute_f1_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute F1 scores for CSI prediction using PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        
    Returns:
        Dictionary with F1 metrics
    """
    # IMPORTANT: Use consistent ignore_index strategy 
    # For medical CSI scores, we recommend ignore_index=4 to focus on gradable cases
    # Change to ignore_index=None if you want to include unknown class in evaluation
    
    # Get both macro and weighted F1 scores for comprehensive analysis
    enhanced_metrics = compute_enhanced_f1_metrics(
        predictions, targets, 
        ignore_index=4,  # Focus on gradable cases for medical interpretation
        include_per_class=False  # Skip per-class details during training for speed
    )
    
    # Return key metrics with backwards compatibility
    return {
        'f1_macro': enhanced_metrics['f1_macro'],
        'f1_weighted_macro': enhanced_metrics['f1_weighted_macro'],
        'f1_overall': enhanced_metrics['f1_overall'],
        'f1_overall_weighted': enhanced_metrics['f1_overall_weighted'],
        'f1_overall_micro': enhanced_metrics['f1_overall_micro'],
        # Keep individual zone metrics
        **{k: v for k, v in enhanced_metrics.items() if k.startswith('f1_zone_')}
    }


def compute_precision_recall(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute precision and recall metrics for CSI prediction using PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        
    Returns:
        Dictionary with precision and recall metrics
    """
    # Note: Now we include all classes (0-4) in precision/recall computation
    return compute_precision_recall_metrics(predictions, targets, ignore_index=None)


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
    
    # Compute F1 metrics and accuracy
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    f1_metrics = compute_f1_metrics(all_pred_tensor, all_target_tensor)
    pr_metrics = compute_precision_recall(all_pred_tensor, all_target_tensor)
    
    # Compute accuracy
    pred_classes = torch.argmax(all_pred_tensor, dim=-1)
    valid_mask = all_target_tensor != 4  # Exclude ungradable samples
    accuracy = (pred_classes[valid_mask] == all_target_tensor[valid_mask]).float().mean().item()
    
    # Combine metrics
    train_metrics = metrics.get_averages()
    train_metrics.update(f1_metrics)
    train_metrics.update(pr_metrics)
    train_metrics['accuracy'] = accuracy
    
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
    
    # Compute F1 metrics and accuracy
    all_pred_tensor = torch.cat(all_predictions, dim=0)
    all_target_tensor = torch.cat(all_targets, dim=0)
    f1_metrics = compute_f1_metrics(all_pred_tensor, all_target_tensor)
    pr_metrics = compute_precision_recall(all_pred_tensor, all_target_tensor)
    
    # Compute accuracy
    pred_classes = torch.argmax(all_pred_tensor, dim=-1)
    valid_mask = all_target_tensor != 4  # Exclude ungradable samples
    accuracy = (pred_classes[valid_mask] == all_target_tensor[valid_mask]).float().mean().item()
    
    # Combine metrics
    val_metrics = metrics.get_averages()
    val_metrics.update(f1_metrics)
    val_metrics.update(pr_metrics)
    val_metrics['accuracy'] = accuracy
    
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
    
    # Create loss function (weighted cross-entropy)
    criterion = WeightedCSILoss(unknown_weight=0.3)  # 30% weight for unknown class
    criterion = criterion.to(device)  # Move criterion to same device as model
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.patience)
    
    # Generate consistent names for this training run
    train_model_name = make_model_name(config)
    run_name = make_run_name(train_model_name, task_tag="Tr")
    logger.info(f"Training model name: {train_model_name}")
    logger.info(f"Training run name: {run_name}")
    
    # Initialize wandb
    use_wandb = False
    try:
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
    
    # Track metrics for plotting curves
    train_losses = []
    val_losses = []
    train_f1_scores = []
    val_f1_scores = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(1, config.n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{config.n_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
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
        
        # Log metrics
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Log to wandb
        if use_wandb:
            wandb_log = {
                "epoch": epoch,
                "learning_rate": current_lr,
                # Loss metrics
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                # Accuracy metrics
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                # F1 metrics - Enhanced with weighted versions
                "train_f1_macro": train_metrics["f1_macro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "train_f1_weighted": train_metrics.get("f1_weighted_macro", 0),
                "val_f1_weighted": val_metrics.get("f1_weighted_macro", 0),
                "train_f1_overall": train_metrics["f1_overall"],
                "val_f1_overall": val_metrics["f1_overall"],
                "train_f1_overall_weighted": train_metrics.get("f1_overall_weighted", 0),
                "val_f1_overall_weighted": val_metrics.get("f1_overall_weighted", 0),
                # Precision metrics
                "train_precision_macro": train_metrics["precision_macro"],
                "val_precision_macro": val_metrics["precision_macro"],
                "train_precision_overall": train_metrics["precision_overall"],
                "val_precision_overall": val_metrics["precision_overall"],
                # Recall metrics
                "train_recall_macro": train_metrics["recall_macro"],
                "val_recall_macro": val_metrics["recall_macro"],
                "train_recall_overall": train_metrics["recall_overall"],
                "val_recall_overall": val_metrics["recall_overall"]
            }
            
            # Add per-zone metrics
            for i in range(6):
                zone_name = f"zone_{i+1}"
                wandb_log[f"train_f1_{zone_name}"] = train_metrics[f"f1_{zone_name}"]
                wandb_log[f"val_f1_{zone_name}"] = val_metrics[f"f1_{zone_name}"]
                wandb_log[f"train_precision_{zone_name}"] = train_metrics[f"precision_{zone_name}"]
                wandb_log[f"val_precision_{zone_name}"] = val_metrics[f"precision_{zone_name}"]
                wandb_log[f"train_recall_{zone_name}"] = train_metrics[f"recall_{zone_name}"]
                wandb_log[f"val_recall_{zone_name}"] = val_metrics[f"recall_{zone_name}"]
            
            wandb.log(wandb_log)
        
        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1_macro"]
            
            # Use consistent model name
            save_path = Path(config.get_model_path(train_model_name))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics["loss"],
                'val_f1_macro': val_metrics["f1_macro"],
                'config': config,
                'model_name': train_model_name,
                'run_name': run_name if use_wandb else None
            }, save_path)
            
            logger.info(f"Saved best model: {train_model_name}")
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
    
    # Generate training curves
    if len(train_losses) > 0:
        logger.info("Generating training curves...")
        graphs_dir = Path(config.graph_dir) / train_model_name / "training_curves"
        plot_training_curves(
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            train_f1_scores, val_f1_scores,
            str(graphs_dir), run_name
        )
    
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