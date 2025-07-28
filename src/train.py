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
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

from .config import cfg, get_config, copy_config_on_training_start
from .data import create_data_loaders
from .models import build_model
from .utils import EarlyStopping, MetricsTracker, logger, seed_everything, make_run_name, make_model_name, log_config, plot_training_curves, plot_training_curves_grid, create_summary_dashboard, save_training_history
from .metrics import compute_pytorch_f1_metrics, compute_precision_recall_metrics, compute_enhanced_f1_metrics
from .discord_notifier import send_training_notification


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


def compute_csi_average_metrics(predictions: torch.Tensor, targets: torch.Tensor, file_ids: Optional[list] = None, csv_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Compute CSI average metrics by comparing predicted averages with ground truth averages from CSV.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        file_ids: List of file IDs for this batch (optional)
        csv_data: DataFrame containing the original CSV data with 'csi' column (optional)
        
    Returns:
        Dictionary with CSI average metrics
    """
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)  # [batch_size, n_zones]
    
    # Calculate predicted CSI averages (excluding unknown class 4)
    pred_averages = []
    target_averages = []
    
    for i in range(pred_classes.shape[0]):
        # Get predictions and targets for this sample
        sample_preds = pred_classes[i]  # [n_zones]
        sample_targets = targets[i]     # [n_zones]
        
        # Calculate average excluding unknown class (4)
        pred_valid_mask = sample_preds != 4
        target_valid_mask = sample_targets != 4
        
        if pred_valid_mask.sum() > 0:
            pred_avg = sample_preds[pred_valid_mask].float().mean().item()
        else:
            pred_avg = 0.0  # Default if all zones are unknown
            
        if target_valid_mask.sum() > 0:
            target_avg = sample_targets[target_valid_mask].float().mean().item()
        else:
            target_avg = 0.0  # Default if all zones are unknown
            
        pred_averages.append(pred_avg)
        target_averages.append(target_avg)
    
    # Convert to tensors for easier computation
    pred_avg_tensor = torch.tensor(pred_averages)
    target_avg_tensor = torch.tensor(target_averages)
    
    # Basic metrics
    metrics = {
        'csi_avg_pred_mean': pred_avg_tensor.mean().item(),
        'csi_avg_target_mean': target_avg_tensor.mean().item(),
        'csi_avg_mae': torch.abs(pred_avg_tensor - target_avg_tensor).mean().item(),
        'csi_avg_rmse': torch.sqrt(torch.mean((pred_avg_tensor - target_avg_tensor) ** 2)).item(),
        'csi_avg_correlation': torch.corrcoef(torch.stack([pred_avg_tensor, target_avg_tensor]))[0, 1].item() if len(pred_avg_tensor) > 1 else 0.0
    }
    
    # Compare with CSV ground truth if available
    if file_ids is not None and csv_data is not None:
        csv_averages = []
        valid_file_ids = []
        
        for file_id in file_ids:
            if file_id in csv_data['FileID'].values:
                csv_row = csv_data[csv_data['FileID'] == file_id].iloc[0]
                if 'csi' in csv_row and pd.notna(csv_row['csi']):
                    csv_averages.append(csv_row['csi'])
                    valid_file_ids.append(file_id)
        
        if csv_averages:
            csv_avg_tensor = torch.tensor(csv_averages)
            # Get corresponding predicted averages for these file IDs
            csv_pred_averages = []
            for file_id in valid_file_ids:
                if file_id in file_ids:
                    idx = file_ids.index(file_id)
                    csv_pred_averages.append(pred_averages[idx])
            
            if csv_pred_averages:
                csv_pred_avg_tensor = torch.tensor(csv_pred_averages)
                
                metrics.update({
                    'csi_csv_avg_pred_mean': csv_pred_avg_tensor.mean().item(),
                    'csi_csv_avg_target_mean': csv_avg_tensor.mean().item(),
                    'csi_csv_avg_mae': torch.abs(csv_pred_avg_tensor - csv_avg_tensor).mean().item(),
                    'csi_csv_avg_rmse': torch.sqrt(torch.mean((csv_pred_avg_tensor - csv_avg_tensor) ** 2)).item(),
                    'csi_csv_avg_correlation': torch.corrcoef(torch.stack([csv_pred_avg_tensor, csv_avg_tensor]))[0, 1].item() if len(csv_pred_avg_tensor) > 1 else 0.0
                })
    
    return metrics


def compute_ahf_classification_metrics(predictions: torch.Tensor, targets: torch.Tensor, file_ids: Optional[list] = None, csv_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute AHF (Acute Heart Failure) classification metrics based on CSI averages.
    
    AHF Classification:
    - AHF_Class = 0: avg CSI <= 1.3 (Low risk)
    - AHF_Class = 1: 1.3 < avg CSI <= 2.2 (Medium risk)  
    - AHF_Class = 2: avg CSI > 2.2 (High risk)
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        file_ids: List of file IDs for this batch (optional)
        csv_data: DataFrame containing the original CSV data with 'csi' column (optional)
        
    Returns:
        Dictionary with AHF classification metrics and confusion matrices
    """
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)  # [batch_size, n_zones]
    
    def calculate_ahf_class(avg_csi: float) -> int:
        """Calculate AHF class based on average CSI."""
        if avg_csi <= 1.3:
            return 0
        elif avg_csi <= 2.2:
            return 1
        else:
            return 2
    
    # Calculate predicted CSI averages and AHF classes
    pred_ahf_classes = []
    target_ahf_classes = []
    
    for i in range(pred_classes.shape[0]):
        # Get predictions and targets for this sample
        sample_preds = pred_classes[i]  # [n_zones]
        sample_targets = targets[i]     # [n_zones]
        
        # Calculate average excluding unknown class (4)
        pred_valid_mask = sample_preds != 4
        target_valid_mask = sample_targets != 4
        
        if pred_valid_mask.sum() > 0:
            pred_avg = sample_preds[pred_valid_mask].float().mean().item()
        else:
            pred_avg = 0.0  # Default if all zones are unknown
            
        if target_valid_mask.sum() > 0:
            target_avg = sample_targets[target_valid_mask].float().mean().item()
        else:
            target_avg = 0.0  # Default if all zones are unknown
            
        # Calculate AHF classes
        pred_ahf_class = calculate_ahf_class(pred_avg)
        target_ahf_class = calculate_ahf_class(target_avg)
        
        pred_ahf_classes.append(pred_ahf_class)
        target_ahf_classes.append(target_ahf_class)
    
    # Convert to tensors
    pred_ahf_tensor = torch.tensor(pred_ahf_classes)
    target_ahf_tensor = torch.tensor(target_ahf_classes)
    
    # Calculate confusion matrix
    import numpy as np
    
    cm = confusion_matrix(target_ahf_tensor, pred_ahf_tensor, labels=[0, 1, 2])
    
    # Calculate metrics
    ahf_accuracy = accuracy_score(target_ahf_tensor, pred_ahf_tensor)
    ahf_f1_macro = f1_score(target_ahf_tensor, pred_ahf_tensor, average='macro')
    ahf_f1_weighted = f1_score(target_ahf_tensor, pred_ahf_tensor, average='weighted')
    
    # Per-class F1 scores
    ahf_f1_per_class = f1_score(target_ahf_tensor, pred_ahf_tensor, average=None, labels=[0, 1, 2])
    
    metrics = {
        'ahf_accuracy': ahf_accuracy,
        'ahf_f1_macro': ahf_f1_macro,
        'ahf_f1_weighted': ahf_f1_weighted,
        'ahf_f1_class_0': ahf_f1_per_class[0],  # Low risk
        'ahf_f1_class_1': ahf_f1_per_class[1],  # Medium risk
        'ahf_f1_class_2': ahf_f1_per_class[2],  # High risk
        'ahf_confusion_matrix': cm,
        'ahf_pred_classes': pred_ahf_classes,
        'ahf_target_classes': target_ahf_classes
    }
    
    # Compare with CSV ground truth if available
    if file_ids is not None and csv_data is not None:
        csv_ahf_classes = []
        csv_pred_ahf_classes = []
        valid_file_ids = []
        
        for file_id in file_ids:
            if file_id in csv_data['FileID'].values:
                csv_row = csv_data[csv_data['FileID'] == file_id].iloc[0]
                if 'csi' in csv_row and pd.notna(csv_row['csi']):
                    csv_avg_csi = csv_row['csi']
                    csv_ahf_class = calculate_ahf_class(csv_avg_csi)
                    csv_ahf_classes.append(csv_ahf_class)
                    valid_file_ids.append(file_id)
        
        if csv_ahf_classes:
            # Get corresponding predicted AHF classes for these file IDs
            for file_id in valid_file_ids:
                if file_id in file_ids:
                    idx = file_ids.index(file_id)
                    csv_pred_ahf_classes.append(pred_ahf_classes[idx])
            
            if csv_pred_ahf_classes:
                csv_ahf_tensor = torch.tensor(csv_ahf_classes)
                csv_pred_ahf_tensor = torch.tensor(csv_pred_ahf_classes)
                
                # Calculate CSV-based confusion matrix
                csv_cm = confusion_matrix(csv_ahf_tensor, csv_pred_ahf_tensor, labels=[0, 1, 2])
                
                # Calculate CSV-based metrics
                csv_ahf_accuracy = accuracy_score(csv_ahf_tensor, csv_pred_ahf_tensor)
                csv_ahf_f1_macro = f1_score(csv_ahf_tensor, csv_pred_ahf_tensor, average='macro')
                csv_ahf_f1_weighted = f1_score(csv_ahf_tensor, csv_pred_ahf_tensor, average='weighted')
                csv_ahf_f1_per_class = f1_score(csv_ahf_tensor, csv_pred_ahf_tensor, average=None, labels=[0, 1, 2])
                
                metrics.update({
                    'ahf_csv_accuracy': csv_ahf_accuracy,
                    'ahf_csv_f1_macro': csv_ahf_f1_macro,
                    'ahf_csv_f1_weighted': csv_ahf_f1_weighted,
                    'ahf_csv_f1_class_0': csv_ahf_f1_per_class[0],
                    'ahf_csv_f1_class_1': csv_ahf_f1_per_class[1],
                    'ahf_csv_f1_class_2': csv_ahf_f1_per_class[2],
                    'ahf_csv_confusion_matrix': csv_cm,
                    'ahf_csv_pred_classes': csv_pred_ahf_classes,
                    'ahf_csv_target_classes': csv_ahf_classes
                })
    
    return metrics


def save_ahf_confusion_matrices(train_metrics: Dict[str, Any], val_metrics: Dict[str, Any], 
                               save_dir: str, run_name: str) -> None:
    """
    Save AHF confusion matrices as plots.
    
    Args:
        train_metrics: Training metrics containing AHF confusion matrices
        val_metrics: Validation metrics containing AHF confusion matrices
        save_dir: Directory to save the plots
        run_name: Name of the training run
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    ahf_class_names = ['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'AHF Classification Confusion Matrices - {run_name}', fontsize=16, fontweight='bold')
    
    # Training confusion matrix (internal comparison)
    if 'ahf_confusion_matrix' in train_metrics:
        cm_train = train_metrics['ahf_confusion_matrix']
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=ahf_class_names, yticklabels=ahf_class_names,
                   ax=axes[0, 0])
        axes[0, 0].set_title(f'Training AHF Classification\nAccuracy: {train_metrics["ahf_accuracy"]:.3f}, F1: {train_metrics["ahf_f1_macro"]:.3f}')
        axes[0, 0].set_xlabel('Predicted AHF Class')
        axes[0, 0].set_ylabel('True AHF Class')
    
    # Validation confusion matrix (internal comparison)
    if 'ahf_confusion_matrix' in val_metrics:
        cm_val = val_metrics['ahf_confusion_matrix']
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=ahf_class_names, yticklabels=ahf_class_names,
                   ax=axes[0, 1])
        axes[0, 1].set_title(f'Validation AHF Classification\nAccuracy: {val_metrics["ahf_accuracy"]:.3f}, F1: {val_metrics["ahf_f1_macro"]:.3f}')
        axes[0, 1].set_xlabel('Predicted AHF Class')
        axes[0, 1].set_ylabel('True AHF Class')
    
    # Training confusion matrix (CSV comparison)
    if 'ahf_csv_confusion_matrix' in train_metrics:
        cm_train_csv = train_metrics['ahf_csv_confusion_matrix']
        sns.heatmap(cm_train_csv, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=ahf_class_names, yticklabels=ahf_class_names,
                   ax=axes[1, 0])
        axes[1, 0].set_title(f'Training AHF Classification (vs CSV)\nAccuracy: {train_metrics["ahf_csv_accuracy"]:.3f}, F1: {train_metrics["ahf_csv_f1_macro"]:.3f}')
        axes[1, 0].set_xlabel('Predicted AHF Class')
        axes[1, 0].set_ylabel('CSV AHF Class')
    
    # Validation confusion matrix (CSV comparison)
    if 'ahf_csv_confusion_matrix' in val_metrics:
        cm_val_csv = val_metrics['ahf_csv_confusion_matrix']
        sns.heatmap(cm_val_csv, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=ahf_class_names, yticklabels=ahf_class_names,
                   ax=axes[1, 1])
        axes[1, 1].set_title(f'Validation AHF Classification (vs CSV)\nAccuracy: {val_metrics["ahf_csv_accuracy"]:.3f}, F1: {val_metrics["ahf_csv_f1_macro"]:.3f}')
        axes[1, 1].set_xlabel('Predicted AHF Class')
        axes[1, 1].set_ylabel('CSV AHF Class')
    
    plt.tight_layout()
    plt.savefig(save_path / f'ahf_confusion_matrices_{run_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save individual confusion matrices
    if 'ahf_confusion_matrix' in train_metrics:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=ahf_class_names, yticklabels=ahf_class_names)
        plt.title(f'Training AHF Classification - {run_name}\nAccuracy: {train_metrics["ahf_accuracy"]:.3f}, F1: {train_metrics["ahf_f1_macro"]:.3f}')
        plt.xlabel('Predicted AHF Class')
        plt.ylabel('True AHF Class')
        plt.tight_layout()
        plt.savefig(save_path / f'ahf_confusion_matrix_train_{run_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    if 'ahf_confusion_matrix' in val_metrics:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=ahf_class_names, yticklabels=ahf_class_names)
        plt.title(f'Validation AHF Classification - {run_name}\nAccuracy: {val_metrics["ahf_accuracy"]:.3f}, F1: {val_metrics["ahf_f1_macro"]:.3f}')
        plt.xlabel('Predicted AHF Class')
        plt.ylabel('True AHF Class')
        plt.tight_layout()
        plt.savefig(save_path / f'ahf_confusion_matrix_val_{run_name}.png', dpi=300, bbox_inches='tight')
        plt.close()


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    csv_data: Optional[pd.DataFrame] = None
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
    csv_data: Optional[pd.DataFrame] = None
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
    all_file_ids = []
    
    # Check if model supports zone masking
    is_zone_masking_model = hasattr(model, 'ZONE_MAPPING')
    
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validation"):
            # Handle both old and new data formats
            if len(batch_data) == 3:  # New format: (images, targets, file_ids)
                images, targets, file_ids = batch_data
            else:  # Old format: (images, targets)
                images, targets = batch_data
                file_ids = None
            
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            # Use zone masking if model supports it and file_ids are available
            if is_zone_masking_model and file_ids is not None:
                outputs = model(images, file_ids)
            else:
                outputs = model(images)
            
            loss = criterion(outputs, targets)
            
            # Track metrics
            metrics.update("loss", loss.item())
            
            # Store for F1 computation
            all_predictions.append(outputs)
            all_targets.append(targets)
            if file_ids is not None:
                all_file_ids.extend(file_ids)
    
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
    
    # Load CSV data for CSI average comparison
    csv_data = None
    try:
        from .data import load_csv_data, convert_nans_to_unknown
        csv_path = os.path.join(config.csv_dir, config.labels_csv)
        csv_data = load_csv_data(csv_path)
        csv_data = convert_nans_to_unknown(csv_data)
        logger.info(f"Loaded CSV data with {len(csv_data)} samples for CSI average comparison")
    except Exception as e:
        logger.warning(f"Could not load CSV data for CSI average comparison: {e}")
        logger.info("CSI average metrics will be computed without CSV ground truth comparison")
    
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
    train_precisions = []
    val_precisions = []
    
    # Track zone-specific metrics for grid plots
    train_zone_accuracies = {f"zone_{i+1}": [] for i in range(6)}
    val_zone_accuracies = {f"zone_{i+1}": [] for i in range(6)}
    train_zone_f1_scores = {f"zone_{i+1}": [] for i in range(6)}
    val_zone_f1_scores = {f"zone_{i+1}": [] for i in range(6)}
    train_zone_precisions = {f"zone_{i+1}": [] for i in range(6)}
    val_zone_precisions = {f"zone_{i+1}": [] for i in range(6)}
    
    for epoch in range(1, config.n_epochs + 1):
        logger.info(f"Starting epoch {epoch}/{config.n_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, csv_data)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device, csv_data)
        
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
        
        # Store zone-specific metrics for grid plots
        for i in range(6):
            zone_key = f"zone_{i+1}"
            train_zone_f1_scores[zone_key].append(train_metrics.get(f"f1_{zone_key}", 0.0))
            val_zone_f1_scores[zone_key].append(val_metrics.get(f"f1_{zone_key}", 0.0))
            train_zone_precisions[zone_key].append(train_metrics.get(f"precision_{zone_key}", 0.0))
            val_zone_precisions[zone_key].append(val_metrics.get(f"precision_{zone_key}", 0.0))
            # Note: Zone-specific accuracy would need to be computed separately if needed
        
        # Log metrics
        logger.info(f"Epoch {epoch}:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Log CSI average metrics
        if 'csi_avg_mae' in train_metrics:
            logger.info(f"  CSI Avg - Train MAE: {train_metrics['csi_avg_mae']:.4f}, Val MAE: {val_metrics['csi_avg_mae']:.4f}")
            logger.info(f"  CSI Avg - Train RMSE: {train_metrics['csi_avg_rmse']:.4f}, Val RMSE: {val_metrics['csi_avg_rmse']:.4f}")
            if 'csi_csv_avg_mae' in train_metrics:
                logger.info(f"  CSI CSV Avg - Train MAE: {train_metrics['csi_csv_avg_mae']:.4f}, Val MAE: {val_metrics['csi_csv_avg_mae']:.4f}")
                logger.info(f"  CSI CSV Avg - Train RMSE: {train_metrics['csi_csv_avg_rmse']:.4f}, Val RMSE: {val_metrics['csi_csv_avg_rmse']:.4f}")
        
        # Log AHF classification metrics
        if 'ahf_accuracy' in train_metrics:
            logger.info(f"  AHF Class - Train Acc: {train_metrics['ahf_accuracy']:.4f}, Val Acc: {val_metrics['ahf_accuracy']:.4f}")
            logger.info(f"  AHF Class - Train F1: {train_metrics['ahf_f1_macro']:.4f}, Val F1: {val_metrics['ahf_f1_macro']:.4f}")
            if 'ahf_csv_accuracy' in train_metrics:
                logger.info(f"  AHF CSV - Train Acc: {train_metrics['ahf_csv_accuracy']:.4f}, Val Acc: {val_metrics['ahf_csv_accuracy']:.4f}")
                logger.info(f"  AHF CSV - Train F1: {train_metrics['ahf_csv_f1_macro']:.4f}, Val F1: {val_metrics['ahf_csv_f1_macro']:.4f}")
        
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
            
            # Add CSI average metrics
            if 'csi_avg_mae' in train_metrics:
                wandb_log.update({
                    "train_csi_avg_mae": train_metrics["csi_avg_mae"],
                    "val_csi_avg_mae": val_metrics["csi_avg_mae"],
                    "train_csi_avg_rmse": train_metrics["csi_avg_rmse"],
                    "val_csi_avg_rmse": val_metrics["csi_avg_rmse"],
                    "train_csi_avg_correlation": train_metrics["csi_avg_correlation"],
                    "val_csi_avg_correlation": val_metrics["csi_avg_correlation"],
                })
                
                if 'csi_csv_avg_mae' in train_metrics:
                    wandb_log.update({
                        "train_csi_csv_avg_mae": train_metrics["csi_csv_avg_mae"],
                        "val_csi_csv_avg_mae": val_metrics["csi_csv_avg_mae"],
                        "train_csi_csv_avg_rmse": train_metrics["csi_csv_avg_rmse"],
                        "val_csi_csv_avg_rmse": val_metrics["csi_csv_avg_rmse"],
                        "train_csi_csv_avg_correlation": train_metrics["csi_csv_avg_correlation"],
                        "val_csi_csv_avg_correlation": val_metrics["csi_csv_avg_correlation"],
                    })
            
            # Add AHF classification metrics
            if 'ahf_accuracy' in train_metrics:
                wandb_log.update({
                    "train_ahf_accuracy": train_metrics["ahf_accuracy"],
                    "val_ahf_accuracy": val_metrics["ahf_accuracy"],
                    "train_ahf_f1_macro": train_metrics["ahf_f1_macro"],
                    "val_ahf_f1_macro": val_metrics["ahf_f1_macro"],
                    "train_ahf_f1_weighted": train_metrics["ahf_f1_weighted"],
                    "val_ahf_f1_weighted": val_metrics["ahf_f1_weighted"],
                    "train_ahf_f1_class_0": train_metrics["ahf_f1_class_0"],
                    "val_ahf_f1_class_0": val_metrics["ahf_f1_class_0"],
                    "train_ahf_f1_class_1": train_metrics["ahf_f1_class_1"],
                    "val_ahf_f1_class_1": val_metrics["ahf_f1_class_1"],
                    "train_ahf_f1_class_2": train_metrics["ahf_f1_class_2"],
                    "val_ahf_f1_class_2": val_metrics["ahf_f1_class_2"],
                })
                
                if 'ahf_csv_accuracy' in train_metrics:
                    wandb_log.update({
                        "train_ahf_csv_accuracy": train_metrics["ahf_csv_accuracy"],
                        "val_ahf_csv_accuracy": val_metrics["ahf_csv_accuracy"],
                        "train_ahf_csv_f1_macro": train_metrics["ahf_csv_f1_macro"],
                        "val_ahf_csv_f1_macro": val_metrics["ahf_csv_f1_macro"],
                        "train_ahf_csv_f1_weighted": train_metrics["ahf_csv_f1_weighted"],
                        "val_ahf_csv_f1_weighted": val_metrics["ahf_csv_f1_weighted"],
                        "train_ahf_csv_f1_class_0": train_metrics["ahf_csv_f1_class_0"],
                        "val_ahf_csv_f1_class_0": val_metrics["ahf_csv_f1_class_0"],
                        "train_ahf_csv_f1_class_1": train_metrics["ahf_csv_f1_class_1"],
                        "val_ahf_csv_f1_class_1": val_metrics["ahf_csv_f1_class_1"],
                        "train_ahf_csv_f1_class_2": train_metrics["ahf_csv_f1_class_2"],
                        "val_ahf_csv_f1_class_2": val_metrics["ahf_csv_f1_class_2"],
                    })
            
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
    
    # Generate training curves and visualizations
    if len(train_losses) > 0:
        logger.info("Generating comprehensive training visualizations...")
        graphs_dir = Path(config.graph_dir) / train_model_name
        epochs_list = list(range(1, len(train_losses) + 1))
        
        # 1. Main training curves (2x2 grid)
        plot_training_curves(
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            train_f1_scores, val_f1_scores,
            str(graphs_dir / "training_curves"), run_name,
            train_precisions, val_precisions
        )
        
        # 2. Zone-specific training curves grids
        logger.info("Generating zone-specific training curves...")
        
        # F1 Score grid
        plot_training_curves_grid(
            train_zone_f1_scores, val_zone_f1_scores,
            "f1_score", str(graphs_dir), run_name, epochs_list
        )
        
        # Precision grid
        plot_training_curves_grid(
            train_zone_precisions, val_zone_precisions,
            "precision", str(graphs_dir), run_name, epochs_list
        )
        
        # 6. Save AHF confusion matrices (using final epoch metrics)
        logger.info("Generating AHF confusion matrices...")
        try:
            # Get the final epoch metrics for confusion matrix generation
            final_train_metrics = train_metrics if 'train_metrics' in locals() else {}
            final_val_metrics = val_metrics if 'val_metrics' in locals() else {}
            
            save_ahf_confusion_matrices(
                final_train_metrics, final_val_metrics,
                str(graphs_dir), run_name
            )
            logger.info("AHF confusion matrices saved successfully!")
        except Exception as e:
            logger.warning(f"Could not generate AHF confusion matrices: {e}")
        
        # 3. Save training history for evaluation dashboard
        logger.info("Saving training history...")
        history_path = graphs_dir / "training_history.json"
        save_training_history(
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            train_precisions, val_precisions,
            train_f1_scores, val_f1_scores,
            str(history_path)
        )
        
        # 4. Create summary dashboard (saved directly in model folder)
        logger.info("Generating summary dashboard...")
        create_summary_dashboard(
            train_accuracies, val_accuracies,
            train_losses, val_losses,
            train_precisions, val_precisions,
            train_f1_scores, val_f1_scores,
            None,  # confusion_matrix - not available during training
            None,  # roc_curve_data - not available during training
            str(graphs_dir), run_name, epochs_list,
            evaluation_metrics=None  # No static evaluation metrics during training
        )
        
        # 5. Send Discord notification with training results
        logger.info("Sending Discord notification...")
        dashboard_path = graphs_dir / f"{run_name}_summary_dashboard.png"
        
        # Prepare final metrics for Discord notification
        final_train_metrics = {
            'accuracy': train_accuracies[-1] if train_accuracies else 0,
            'loss': train_losses[-1] if train_losses else 0,
            'f1_macro': train_f1_scores[-1] if train_f1_scores else 0,
            'precision_macro': train_precisions[-1] if train_precisions else 0
        }
        
        final_val_metrics = {
            'accuracy': val_accuracies[-1] if val_accuracies else 0,
            'loss': best_val_loss,
            'f1_macro': best_val_f1,
            'precision_macro': val_precisions[-1] if val_precisions else 0
        }
        
        send_training_notification(
            config=config,
            model_name=train_model_name,
            dashboard_image_path=str(dashboard_path) if dashboard_path.exists() else None,
            train_results=final_train_metrics,
            val_results=final_val_metrics
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