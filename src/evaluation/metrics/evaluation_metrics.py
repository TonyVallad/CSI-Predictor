"""
Evaluation-specific metrics for CSI-Predictor.

This module contains evaluation metrics functionality extracted from the original src/evaluate.py file.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import confusion_matrix, classification_report
from ..metrics.confusion_matrix import compute_confusion_matrix

def compute_confusion_matrices_per_zone(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrices for each CSI zone using pure PyTorch.
    
    Args:
        predictions: Predicted CSI class indices [num_samples, num_zones]
        targets: Ground truth CSI class indices [num_samples, num_zones]
        
    Returns:
        Dictionary of confusion matrices per zone
    """
    zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    confusion_matrices = {}
    
    for i, zone_name in enumerate(zone_names):
        # Get all samples for this zone (including unknown class 4)
        zone_pred = predictions[:, i]
        zone_true = targets[:, i]
        
        if len(zone_pred) > 0:
            # Compute confusion matrix using sklearn
            cm = confusion_matrix(zone_true, zone_pred, labels=[0, 1, 2, 3, 4])
            confusion_matrices[zone_name] = cm
        else:
            # Empty confusion matrix if no samples
            confusion_matrices[zone_name] = np.zeros((5, 5), dtype=int)
    
    return confusion_matrices


def create_classification_report_per_zone(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict]:
    """
    Create classification report for each CSI zone.
    
    Args:
        predictions: Predicted CSI class indices [num_samples, num_zones]
        targets: Ground truth CSI class indices [num_samples, num_zones]
        
    Returns:
        Dictionary of classification reports per zone
    """
    zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    reports = {}
    
    for i, zone_name in enumerate(zone_names):
        # Get all samples for this zone
        zone_pred = predictions[:, i]
        zone_true = targets[:, i]
        
        if len(zone_pred) > 0:
            # Create classification report using sklearn
            report = classification_report(
                zone_true, zone_pred, 
                labels=[0, 1, 2, 3, 4],
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            reports[zone_name] = report
        else:
            # Empty report if no samples
            reports[zone_name] = {
                'accuracy': 0.0,
                'macro avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0},
                'weighted avg': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0}
            }
    
    return reports


def compute_zone_metrics(predictions: np.ndarray, targets: np.ndarray, zone_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each CSI zone (classification version).
    
    Args:
        predictions: Predicted CSI class indices [num_samples, num_zones]
        targets: Ground truth CSI class indices [num_samples, num_zones]
        zone_names: Names of CSI zones
        
    Returns:
        Dictionary of metrics per zone
    """
    # Convert numpy arrays to PyTorch tensors
    import torch
    pred_tensor = torch.from_numpy(predictions)
    target_tensor = torch.from_numpy(targets)
    
    # Create dummy logits tensor for the PyTorch metrics function
    # Shape: [num_samples, num_zones, num_classes]
    batch_size, num_zones = pred_tensor.shape
    num_classes = 5
    
    # Create one-hot style logits where the predicted class has highest value
    logits = torch.zeros(batch_size, num_zones, num_classes)
    for i in range(batch_size):
        for j in range(num_zones):
            pred_class = pred_tensor[i, j]
            if pred_class < num_classes:  # Valid class
                logits[i, j, pred_class] = 1.0
    
    # Use our PyTorch metrics
    from .f1_score import compute_pytorch_f1_metrics
    from .classification import compute_accuracy
    
    f1_metrics = compute_pytorch_f1_metrics(logits, target_tensor, ignore_index=4)
    accuracy_metrics = compute_accuracy(logits, target_tensor, ignore_index=4)
    
    # Map zone names to metric keys (zone_1, zone_2, etc.)
    zone_key_mapping = {
        "right_sup": "zone_1",
        "left_sup": "zone_2", 
        "right_mid": "zone_3",
        "left_mid": "zone_4",
        "right_inf": "zone_5",
        "left_inf": "zone_6"
    }
    
    # Reorganize into per-zone format
    zone_metrics = {}
    for zone_name in zone_names:
        metric_key = zone_key_mapping.get(zone_name, f"zone_{zone_names.index(zone_name) + 1}")
        zone_idx = zone_names.index(zone_name)
        
        zone_metrics[zone_name] = {
            'f1_macro': f1_metrics.get(f'f1_{metric_key}', 0.0),
            'f1_weighted': f1_metrics.get(f'f1_{metric_key}', 0.0),  # Use same as macro for simplicity
            'accuracy': accuracy_metrics.get(f'acc_{metric_key}', 0.0),
            'valid_samples': int((targets[:, zone_idx] != 4).sum())
        }
    
    return zone_metrics


def compute_overall_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute overall metrics across all zones.
    
    Args:
        predictions: Predicted CSI class indices [num_samples, num_zones]
        targets: Ground truth CSI class indices [num_samples, num_zones]
        
    Returns:
        Dictionary of overall metrics
    """
    # Convert numpy arrays to PyTorch tensors
    import torch
    pred_tensor = torch.from_numpy(predictions)
    target_tensor = torch.from_numpy(targets)
    
    # Create dummy logits tensor for the PyTorch metrics function
    batch_size, num_zones = pred_tensor.shape
    num_classes = 5
    
    # Create one-hot style logits where the predicted class has highest value
    logits = torch.zeros(batch_size, num_zones, num_classes)
    for i in range(batch_size):
        for j in range(num_zones):
            pred_class = pred_tensor[i, j]
            if pred_class < num_classes:  # Valid class
                logits[i, j, pred_class] = 1.0
    
    # Use our PyTorch metrics
    from .f1_score import compute_pytorch_f1_metrics
    from .classification import compute_accuracy, compute_precision_recall_metrics
    
    f1_metrics = compute_pytorch_f1_metrics(logits, target_tensor, ignore_index=4)
    accuracy_metrics = compute_accuracy(logits, target_tensor, ignore_index=4)
    pr_metrics = compute_precision_recall_metrics(logits, target_tensor, ignore_index=4)
    
    # Extract overall metrics
    overall_metrics = {
        'f1_macro': f1_metrics.get('f1_macro', 0.0),
        'f1_weighted': f1_metrics.get('f1_weighted_macro', 0.0),
        'f1_overall': f1_metrics.get('f1_overall', 0.0),
        'accuracy': accuracy_metrics.get('accuracy', 0.0),
        'precision_macro': pr_metrics.get('precision_macro', 0.0),
        'recall_macro': pr_metrics.get('recall_macro', 0.0),
        'precision_overall': pr_metrics.get('precision_overall', 0.0),
        'recall_overall': pr_metrics.get('recall_overall', 0.0),
        'total_samples': len(predictions),
        'valid_samples': int((targets != 4).sum())
    }
    
    return overall_metrics


def create_overall_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Create an overall confusion matrix by combining all zones.
    
    Args:
        predictions: Predicted CSI class indices [num_samples, num_zones]
        targets: Ground truth CSI class indices [num_samples, num_zones]
        
    Returns:
        Overall confusion matrix [num_classes, num_classes]
    """
    from sklearn.metrics import confusion_matrix
    
    # Flatten predictions and targets across all zones
    all_predictions = predictions.flatten()
    all_targets = targets.flatten()
    
    # Compute overall confusion matrix
    overall_cm = confusion_matrix(all_targets, all_predictions, labels=[0, 1, 2, 3, 4])
    
    return overall_cm


__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 