"""
Classification metrics for CSI-Predictor.

This module contains classification metrics functionality extracted from the original src/metrics.py file.
"""

import torch
from typing import Dict, List, Optional

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = 4) -> Dict[str, float]:
    """
    Compute accuracy metrics using PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        ignore_index: Class index to ignore (None to include all classes)
        
    Returns:
        Dictionary with accuracy metrics
    """
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)
    
    # Create mask for valid targets
    if ignore_index is not None:
        mask = (targets != ignore_index)
    else:
        # Include all samples when ignore_index is None
        mask = torch.ones_like(targets, dtype=torch.bool)
    
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


def compute_precision_recall_metrics(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = 4) -> Dict[str, float]:
    """
    Compute precision and recall metrics using PyTorch.
    
    Args:
        predictions: Model predictions [batch_size, n_zones, n_classes]
        targets: Ground truth labels [batch_size, n_zones]
        ignore_index: Class index to ignore (None to include all classes)
        
    Returns:
        Dictionary with precision and recall metrics
    """
    from .confusion_matrix import compute_confusion_matrix
    
    # Convert predictions to class indices
    pred_classes = torch.argmax(predictions, dim=-1)
    
    # Create mask for valid targets
    if ignore_index is not None:
        mask = (targets != ignore_index)
    else:
        # Include all samples when ignore_index is None
        mask = torch.ones_like(targets, dtype=torch.bool)
    
    metrics = {}
    zone_names = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "zone_6"]
    
    # Per-zone precision and recall
    zone_precisions = []
    zone_recalls = []
    
    for i, zone_name in enumerate(zone_names):
        zone_mask = mask[:, i]
        if zone_mask.sum() > 0:
            zone_pred = pred_classes[:, i][zone_mask]
            zone_true = targets[:, i][zone_mask]
            
            # Compute confusion matrix for this zone
            cm = compute_confusion_matrix(zone_pred, zone_true, num_classes=5)
            
            # Compute precision and recall per class, then average
            tp = torch.diag(cm)
            fp = cm.sum(dim=0) - tp
            fn = cm.sum(dim=1) - tp
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            # Macro average for this zone
            zone_precision = precision.mean().item()
            zone_recall = recall.mean().item()
            
            metrics[f"precision_{zone_name}"] = zone_precision
            metrics[f"recall_{zone_name}"] = zone_recall
            
            zone_precisions.append(zone_precision)
            zone_recalls.append(zone_recall)
        else:
            metrics[f"precision_{zone_name}"] = 0.0
            metrics[f"recall_{zone_name}"] = 0.0
            zone_precisions.append(0.0)
            zone_recalls.append(0.0)
    
    # Overall precision and recall
    metrics["precision_macro"] = sum(zone_precisions) / len(zone_precisions)
    metrics["recall_macro"] = sum(zone_recalls) / len(zone_recalls)
    
    # Overall precision and recall (all zones flattened)
    if mask.sum() > 0:
        all_pred = pred_classes[mask]
        all_true = targets[mask]
        
        # Compute overall confusion matrix
        cm_overall = compute_confusion_matrix(all_pred, all_true, num_classes=5)
        tp_overall = torch.diag(cm_overall)
        fp_overall = cm_overall.sum(dim=0) - tp_overall
        fn_overall = cm_overall.sum(dim=1) - tp_overall
        
        precision_overall = tp_overall / (tp_overall + fp_overall + 1e-8)
        recall_overall = tp_overall / (tp_overall + fn_overall + 1e-8)
        
        metrics["precision_overall"] = precision_overall.mean().item()
        metrics["recall_overall"] = recall_overall.mean().item()
    else:
        metrics["precision_overall"] = 0.0
        metrics["recall_overall"] = 0.0
    
    return metrics

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 