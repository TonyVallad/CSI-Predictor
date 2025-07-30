"""
F1 score calculations for CSI-Predictor.

This module contains F1 score functionality extracted from the original src/metrics.py file.
"""

import torch
from typing import Dict, List, Optional

def compute_f1_from_confusion_matrix(cm: torch.Tensor, average: str = 'macro') -> torch.Tensor:
    """
    Compute F1 score from confusion matrix.
    
    Args:
        cm: Confusion matrix [num_classes, num_classes]
        average: Averaging method ('macro', 'weighted', 'micro')
        
    Returns:
        F1 score scalar tensor
        
    Note:
        For imbalanced datasets (like medical data), consider:
        - 'weighted': Accounts for class frequency
        - 'macro': Treats all classes equally (current default)
        - Per-class F1 scores for detailed analysis
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


def compute_pytorch_f1_metrics(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = 4) -> Dict[str, float]:
    """
    Compute F1 scores for CSI prediction using pure PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        ignore_index: Class index to ignore (default: 4 for ungradable, None to include all classes)
        
    Returns:
        Dictionary with F1 metrics
    """
    from .confusion_matrix import compute_confusion_matrix
    
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)  # [batch_size, n_zones]
    
    # Create mask for valid targets (not equal to ignore_index)
    if ignore_index is not None:
        mask = (targets != ignore_index)
    else:
        # Include all samples when ignore_index is None
        mask = torch.ones_like(targets, dtype=torch.bool)
    
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


def compute_per_class_f1_scores(cm: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class F1 scores from confusion matrix.
    
    Args:
        cm: Confusion matrix [num_classes, num_classes]
        
    Returns:
        Per-class F1 scores [num_classes]
    """
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return f1


def compute_enhanced_f1_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor, 
    ignore_index: Optional[int] = 4,
    include_per_class: bool = True
) -> Dict[str, float]:
    """
    Enhanced F1 metrics computation with detailed per-class analysis.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        ignore_index: Class index to ignore (default: 4 for ungradable, None to include all classes)
        include_per_class: Whether to include per-class F1 scores
        
    Returns:
        Dictionary with comprehensive F1 metrics including per-class scores
    """
    from .confusion_matrix import compute_confusion_matrix
    
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)  # [batch_size, n_zones]
    
    # Create mask for valid targets
    if ignore_index is not None:
        mask = (targets != ignore_index)
    else:
        mask = torch.ones_like(targets, dtype=torch.bool)
    
    f1_metrics = {}
    zone_names = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "zone_6"]
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    # Per-zone F1 scores with detailed per-class metrics
    zone_f1_scores = []
    zone_weighted_f1_scores = []
    
    for i, zone_name in enumerate(zone_names):
        zone_mask = mask[:, i]
        if zone_mask.sum() > 0:  # Only compute if there are valid samples
            zone_pred = pred_classes[:, i][zone_mask]
            zone_true = targets[:, i][zone_mask]
            
            # Compute confusion matrix for this zone
            cm = compute_confusion_matrix(zone_pred, zone_true, num_classes=5)
            
            # Compute different F1 averaging methods
            zone_f1_macro = compute_f1_from_confusion_matrix(cm, average='macro')
            zone_f1_weighted = compute_f1_from_confusion_matrix(cm, average='weighted')
            
            f1_metrics[f"f1_{zone_name}"] = zone_f1_macro.item()
            f1_metrics[f"f1_weighted_{zone_name}"] = zone_f1_weighted.item()
            
            zone_f1_scores.append(zone_f1_macro.item())
            zone_weighted_f1_scores.append(zone_f1_weighted.item())
            
            # Per-class F1 scores for this zone
            if include_per_class:
                per_class_f1 = compute_per_class_f1_scores(cm)
                for class_idx, class_name in enumerate(class_names):
                    f1_metrics[f"f1_{zone_name}_{class_name.lower()}"] = per_class_f1[class_idx].item()
                    
        else:
            f1_metrics[f"f1_{zone_name}"] = 0.0
            f1_metrics[f"f1_weighted_{zone_name}"] = 0.0
            zone_f1_scores.append(0.0)
            zone_weighted_f1_scores.append(0.0)
            
            # Zero per-class F1 scores
            if include_per_class:
                for class_name in class_names:
                    f1_metrics[f"f1_{zone_name}_{class_name.lower()}"] = 0.0
    
    # Macro-averaged F1 across all zones
    f1_metrics["f1_macro"] = sum(zone_f1_scores) / len(zone_f1_scores)
    f1_metrics["f1_weighted_macro"] = sum(zone_weighted_f1_scores) / len(zone_weighted_f1_scores)
    
    # Overall F1 (all zones flattened)
    if mask.sum() > 0:
        all_pred = pred_classes[mask]
        all_true = targets[mask]
        
        # Compute overall confusion matrix
        cm_overall = compute_confusion_matrix(all_pred, all_true, num_classes=5)
        f1_overall = compute_f1_from_confusion_matrix(cm_overall, average='macro')
        f1_weighted_overall = compute_f1_from_confusion_matrix(cm_overall, average='weighted')
        
        f1_metrics["f1_overall"] = f1_overall.item()
        f1_metrics["f1_weighted_overall"] = f1_weighted_overall.item()
        
        # Overall per-class F1 scores
        if include_per_class:
            per_class_f1_overall = compute_per_class_f1_scores(cm_overall)
            for class_idx, class_name in enumerate(class_names):
                f1_metrics[f"f1_overall_{class_name.lower()}"] = per_class_f1_overall[class_idx].item()
    else:
        f1_metrics["f1_overall"] = 0.0
        f1_metrics["f1_weighted_overall"] = 0.0
        if include_per_class:
            for class_name in class_names:
                f1_metrics[f"f1_overall_{class_name.lower()}"] = 0.0
    
    return f1_metrics

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 