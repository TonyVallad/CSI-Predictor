"""
Pure PyTorch metrics for CSI-Predictor.
Replaces scikit-learn metrics with native PyTorch implementations.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """
    Compute confusion matrix using PyTorch.
    
    Args:
        predictions: Predicted class indices [N]
        targets: Ground truth class indices [N]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    # Create confusion matrix using bincount
    indices = num_classes * targets + predictions
    cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm.float()


def compute_f1_from_confusion_matrix(cm: torch.Tensor, average: str = 'macro') -> torch.Tensor:
    """
    Compute F1 score from confusion matrix.
    
    Args:
        cm: Confusion matrix [num_classes, num_classes]
        average: Averaging method ('macro', 'weighted', 'micro')
        
    Returns:
        F1 score scalar tensor
    """
    # True positives, false positives, false negatives
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    
    # Precision and recall
    precision = tp / (tp + fp + 1e-8)  # Add epsilon to avoid division by zero
    recall = tp / (tp + fn + 1e-8)
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Handle averaging
    if average == 'macro':
        return f1.mean()
    elif average == 'weighted':
        support = cm.sum(dim=1)
        return (f1 * support).sum() / support.sum()
    elif average == 'micro':
        # Micro-average: compute metrics globally
        tp_total = tp.sum()
        fp_total = fp.sum()
        fn_total = fn.sum()
        
        precision_micro = tp_total / (tp_total + fp_total + 1e-8)
        recall_micro = tp_total / (tp_total + fn_total + 1e-8)
        
        return 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)
    else:
        raise ValueError(f"Unsupported average method: {average}")


def compute_pytorch_f1_metrics(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = 4) -> Dict[str, float]:
    """
    Compute F1 scores for CSI prediction using pure PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        ignore_index: Class index to ignore (default: 4 for ungradable)
        
    Returns:
        Dictionary with F1 metrics
    """
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)  # [batch_size, n_zones]
    
    # Create mask for valid targets (not equal to ignore_index)
    mask = (targets != ignore_index)
    
    f1_metrics = {}
    zone_names = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "zone_6"]
    
    # Per-zone F1 scores
    zone_f1_scores = []
    for i, zone_name in enumerate(zone_names):
        zone_mask = mask[:, i]
        if zone_mask.sum() > 0:  # Only compute if there are valid samples
            zone_pred = pred_classes[:, i][zone_mask]
            zone_true = targets[:, i][zone_mask]
            
            # Compute confusion matrix for this zone
            cm = compute_confusion_matrix(zone_pred, zone_true, num_classes=5)
            
            # Compute F1 score
            zone_f1 = compute_f1_from_confusion_matrix(cm, average='macro')
            f1_metrics[f"f1_{zone_name}"] = zone_f1.item()
            zone_f1_scores.append(zone_f1.item())
        else:
            f1_metrics[f"f1_{zone_name}"] = 0.0
            zone_f1_scores.append(0.0)
    
    # Macro-averaged F1 across all zones
    f1_metrics["f1_macro"] = sum(zone_f1_scores) / len(zone_f1_scores)
    
    # Overall F1 (all zones flattened)
    if mask.sum() > 0:
        all_pred = pred_classes[mask]
        all_true = targets[mask]
        
        # Compute overall confusion matrix
        cm_overall = compute_confusion_matrix(all_pred, all_true, num_classes=5)
        f1_overall = compute_f1_from_confusion_matrix(cm_overall, average='macro')
        f1_metrics["f1_overall"] = f1_overall.item()
    else:
        f1_metrics["f1_overall"] = 0.0
    
    return f1_metrics


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = 4) -> Dict[str, float]:
    """
    Compute accuracy metrics using PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        ignore_index: Class index to ignore
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)
    
    # Create mask for valid targets
    mask = (targets != ignore_index)
    
    accuracy_metrics = {}
    zone_names = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "zone_6"]
    
    # Per-zone accuracy
    zone_accuracies = []
    for i, zone_name in enumerate(zone_names):
        zone_mask = mask[:, i]
        if zone_mask.sum() > 0:
            zone_pred = pred_classes[:, i][zone_mask]
            zone_true = targets[:, i][zone_mask]
            zone_acc = (zone_pred == zone_true).float().mean()
            accuracy_metrics[f"acc_{zone_name}"] = zone_acc.item()
            zone_accuracies.append(zone_acc.item())
        else:
            accuracy_metrics[f"acc_{zone_name}"] = 0.0
            zone_accuracies.append(0.0)
    
    # Overall accuracy
    if mask.sum() > 0:
        all_pred = pred_classes[mask]
        all_true = targets[mask]
        overall_acc = (all_pred == all_true).float().mean()
        accuracy_metrics["acc_overall"] = overall_acc.item()
        accuracy_metrics["acc_macro"] = sum(zone_accuracies) / len(zone_accuracies)
    else:
        accuracy_metrics["acc_overall"] = 0.0
        accuracy_metrics["acc_macro"] = 0.0
    
    return accuracy_metrics


# Export main functions
__all__ = [
    'compute_pytorch_f1_metrics',
    'compute_accuracy',
    'compute_confusion_matrix',
    'compute_f1_from_confusion_matrix'
] 