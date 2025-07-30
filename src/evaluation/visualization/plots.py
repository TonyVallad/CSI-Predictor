"""
Plotting utilities for CSI-Predictor evaluation.

This module contains plotting functionality extracted from the original src/utils.py file.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
from ...utils.logging import logger

def create_roc_curves(
    predictions_proba: np.ndarray,
    targets: np.ndarray,
    zone_names: List[str],
    class_names: List[str],
    save_dir: str,
    split_name: str = "validation",
    ignore_class: int = 4
) -> Dict[str, Dict]:
    """
    Create ROC curves for multi-class, multi-zone CSI predictions.
    
    Args:
        predictions_proba: Prediction probabilities [N, zones, classes]
        targets: Ground truth labels [N, zones]
        zone_names: Names of anatomical zones
        class_names: Names of CSI classes
        save_dir: Directory to save plots
        split_name: Name of the data split
        ignore_class: Class to ignore in evaluation (default: 4 for ungradable)
        
    Returns:
        Dictionary with ROC metrics per zone and class
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    import matplotlib.colors as mcolors
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create organized subfolders
    overall_dir = save_path / "roc_curves" / "overall"
    zones_dir = save_path / "roc_curves" / "individual_zones"
    overall_dir.mkdir(parents=True, exist_ok=True)
    zones_dir.mkdir(parents=True, exist_ok=True)
    
    roc_metrics = {}
    
    # Color palette for classes
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    
    for zone_idx, zone_name in enumerate(zone_names):
        zone_targets = targets[:, zone_idx]
        zone_proba = predictions_proba[:, zone_idx, :]
        
        # Filter out ignore_class samples
        valid_mask = zone_targets != ignore_class
        if not valid_mask.any():
            logger.warning(f"No valid samples for zone {zone_name} in {split_name} set")
            continue
            
        zone_targets_valid = zone_targets[valid_mask]
        zone_proba_valid = zone_proba[valid_mask]
        
        # Check if we have all classes represented
        unique_classes = np.unique(zone_targets_valid)
        n_classes = len([c for c in unique_classes if c != ignore_class])
        
        if n_classes < 2:
            logger.warning(f"Zone {zone_name} has less than 2 classes, skipping ROC curve")
            continue
        
        # Create a figure for this zone
        plt.figure(figsize=(10, 8))
        
        zone_roc_data = {}
        all_fpr = []
        all_tpr = []
        all_aucs = []
        
        # Filter class_names to exclude ignore_class
        valid_classes = [i for i in range(len(class_names)) if i != ignore_class]
        valid_class_names = [class_names[i] for i in valid_classes if i < len(class_names)]
        
        # Plot ROC curve for each class (One-vs-Rest approach)
        for class_idx in valid_classes:
            if class_idx >= zone_proba_valid.shape[1]:
                continue
                
            # Create binary problem: current class vs all others
            binary_targets = (zone_targets_valid == class_idx).astype(int)
            
            # Skip if this class has no positive samples
            if binary_targets.sum() == 0:
                continue
                
            class_proba = zone_proba_valid[:, class_idx]
            
            # Compute ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(binary_targets, class_proba)
            roc_auc = auc(fpr, tpr)
            
            # Store metrics
            zone_roc_data[class_names[class_idx]] = {
                'fpr': fpr,
                'tpr': tpr,
                'auc': roc_auc,
                'thresholds': thresholds
            }
            
            # Plot ROC curve
            color = colors[class_idx % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_names[class_idx]} (AUC = {roc_auc:.3f})')
            
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            all_aucs.append(roc_auc)
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        
        # Calculate micro-average ROC curve and AUC
        if len(all_fpr) > 0:
            all_fpr = np.concatenate(all_fpr)
            all_tpr = np.concatenate(all_tpr)
            micro_auc = auc(all_fpr, all_tpr)
            plt.plot(all_fpr, all_tpr, color='red', lw=2,
                    label=f'Micro-average (AUC = {micro_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {zone_name} ({split_name})')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = zones_dir / f"roc_curve_{zone_name}_{split_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        roc_metrics[zone_name] = zone_roc_data
    
    return roc_metrics


def create_precision_recall_curves(
    predictions_proba: np.ndarray,
    targets: np.ndarray,
    zone_names: List[str],
    class_names: List[str],
    save_dir: str,
    split_name: str = "validation",
    ignore_class: int = 4
) -> Dict[str, Dict]:
    """
    Create Precision-Recall curves for multi-class, multi-zone CSI predictions.
    
    Args:
        predictions_proba: Prediction probabilities [N, zones, classes]
        targets: Ground truth labels [N, zones]
        zone_names: Names of anatomical zones
        class_names: Names of CSI classes
        save_dir: Directory to save plots
        split_name: Name of the data split
        ignore_class: Class to ignore in evaluation (default: 4 for ungradable)
        
    Returns:
        Dictionary with PR metrics per zone and class
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.colors as mcolors
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create organized subfolders
    overall_dir = save_path / "precision_recall_curves" / "overall"
    zones_dir = save_path / "precision_recall_curves" / "individual_zones"
    overall_dir.mkdir(parents=True, exist_ok=True)
    zones_dir.mkdir(parents=True, exist_ok=True)
    
    pr_metrics = {}
    
    # Color palette for classes
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    
    for zone_idx, zone_name in enumerate(zone_names):
        zone_targets = targets[:, zone_idx]
        zone_proba = predictions_proba[:, zone_idx, :]
        
        # Filter out ignore_class samples
        valid_mask = zone_targets != ignore_class
        if not valid_mask.any():
            logger.warning(f"No valid samples for zone {zone_name} in {split_name} set")
            continue
            
        zone_targets_valid = zone_targets[valid_mask]
        zone_proba_valid = zone_proba[valid_mask]
        
        # Check if we have all classes represented
        unique_classes = np.unique(zone_targets_valid)
        n_classes = len([c for c in unique_classes if c != ignore_class])
        
        if n_classes < 2:
            logger.warning(f"Zone {zone_name} has less than 2 classes, skipping PR curve")
            continue
        
        # Create a figure for this zone
        plt.figure(figsize=(10, 8))
        
        zone_pr_data = {}
        all_precisions = []
        all_recalls = []
        all_aps = []
        
        # Filter class_names to exclude ignore_class
        valid_classes = [i for i in range(len(class_names)) if i != ignore_class]
        valid_class_names = [class_names[i] for i in valid_classes if i < len(class_names)]
        
        # Plot PR curve for each class
        for class_idx in valid_classes:
            if class_idx >= zone_proba_valid.shape[1]:
                continue
                
            # Create binary problem: current class vs all others
            binary_targets = (zone_targets_valid == class_idx).astype(int)
            
            # Skip if this class has no positive samples
            if binary_targets.sum() == 0:
                continue
                
            class_proba = zone_proba_valid[:, class_idx]
            
            # Compute PR curve and AP
            precision, recall, thresholds = precision_recall_curve(binary_targets, class_proba)
            ap = average_precision_score(binary_targets, class_proba)
            
            # Store metrics
            zone_pr_data[class_names[class_idx]] = {
                'precision': precision,
                'recall': recall,
                'ap': ap,
                'thresholds': thresholds
            }
            
            # Plot PR curve
            color = colors[class_idx % len(colors)]
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{class_names[class_idx]} (AP = {ap:.3f})')
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_aps.append(ap)
        
        # Calculate micro-average PR curve
        if len(all_aps) > 0:
            micro_ap = np.mean(all_aps)
            plt.axhline(y=micro_ap, color='red', linestyle='--', lw=2,
                       label=f'Micro-average (AP = {micro_ap:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {zone_name} ({split_name})')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = zones_dir / f"pr_curve_{zone_name}_{split_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        pr_metrics[zone_name] = zone_pr_data
    
    return pr_metrics


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    save_dir: str,
    run_name: str,
    train_precisions: List[float] = None,
    val_precisions: List[float] = None
) -> None:
    """
    Plot training curves for loss, accuracy, and F1 score.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accuracies: Training accuracies per epoch
        val_accuracies: Validation accuracies per epoch
        train_f1_scores: Training F1 scores per epoch
        val_f1_scores: Validation F1 scores per epoch
        save_dir: Directory to save plots
        run_name: Name of the training run
        train_precisions: Training precisions per epoch (optional)
        val_precisions: Validation precisions per epoch (optional)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    epochs = list(range(1, len(train_losses) + 1))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Curves - {run_name}', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[0, 1].plot(epochs, train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, val_accuracies, 'r-', label='Val Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot F1 score
    axes[1, 0].plot(epochs, train_f1_scores, 'b-', label='Train F1', linewidth=2)
    axes[1, 0].plot(epochs, val_f1_scores, 'r-', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot precision if available
    if train_precisions and val_precisions:
        axes[1, 1].plot(epochs, train_precisions, 'b-', label='Train Precision', linewidth=2)
        axes[1, 1].plot(epochs, val_precisions, 'r-', label='Val Precision', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Precision data not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Precision')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path / f"training_curves_{run_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves to {plot_path}")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 