"""
Training metrics for CSI-Predictor.

This module contains training metrics functionality extracted from the original src/train.py file.
"""

import torch
import pandas as pd
from typing import Dict, Any, Optional
from ..evaluation.metrics.f1_score import compute_enhanced_f1_metrics
from ..evaluation.metrics.classification import compute_precision_recall_metrics

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
    
    # Debug logging for metric computation
    f1_weighted_overall = enhanced_metrics.get('f1_weighted_overall', 0.0)
    f1_macro = enhanced_metrics.get('f1_macro', 0.0)
    
    # Check for invalid values
    if torch.isnan(torch.tensor(f1_weighted_overall)) or torch.isinf(torch.tensor(f1_weighted_overall)):
        logger.warning(f"Invalid f1_weighted_overall detected: {f1_weighted_overall}, using 0.0")
        f1_weighted_overall = 0.0
    
    if torch.isnan(torch.tensor(f1_macro)) or torch.isinf(torch.tensor(f1_macro)):
        logger.warning(f"Invalid f1_macro detected: {f1_macro}, using 0.0")
        f1_macro = 0.0
    
    logger.debug(f"Computed metrics - f1_weighted_overall: {f1_weighted_overall}, f1_macro: {f1_macro}")
    
    # Return key metrics with backwards compatibility
    return {
        'f1_macro': f1_macro,
        'f1_weighted_macro': enhanced_metrics.get('f1_weighted_macro', 0.0),
        'f1_overall': enhanced_metrics.get('f1_overall', 0.0),
        'f1_weighted_overall': f1_weighted_overall,  # Consistent key name
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
    
    # Calculate basic metrics
    correct = (pred_ahf_tensor == target_ahf_tensor).sum().item()
    total = len(pred_ahf_tensor)
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate confusion matrix
    from ..evaluation.metrics.confusion_matrix import compute_confusion_matrix
    cm = compute_confusion_matrix(pred_ahf_tensor, target_ahf_tensor, num_classes=3)
    
    # Calculate per-class metrics
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Macro averages
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()
    
    metrics = {
        'ahf_accuracy': accuracy,
        'ahf_macro_precision': macro_precision,
        'ahf_macro_recall': macro_recall,
        'ahf_macro_f1': macro_f1,
        'ahf_class_0_precision': precision[0].item(),
        'ahf_class_0_recall': recall[0].item(),
        'ahf_class_0_f1': f1[0].item(),
        'ahf_class_1_precision': precision[1].item(),
        'ahf_class_1_recall': recall[1].item(),
        'ahf_class_1_f1': f1[1].item(),
        'ahf_class_2_precision': precision[2].item(),
        'ahf_class_2_recall': recall[2].item(),
        'ahf_class_2_f1': f1[2].item(),
        'ahf_confusion_matrix': cm.cpu().numpy().astype(int).tolist()
    }
    
    return metrics

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 