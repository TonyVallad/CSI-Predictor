"""
Main evaluation logic for CSI-Predictor.

This module contains the main evaluation functionality extracted from the original src/evaluate.py file.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

from ..models import build_model, CSIModel
from ..utils.logging import logger
from ..utils.checkpoint import load_checkpoint
from .metrics import compute_confusion_matrices_per_zone, create_classification_report_per_zone
from .visualization import save_confusion_matrix_graphs
from .wandb_logging import log_to_wandb

def load_trained_model(model_path: str, device: torch.device, fallback_config=None):
    """
    Load trained model from checkpoint with automatic architecture detection.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        fallback_config: Fallback configuration if checkpoint doesn't contain config
        
    Returns:
        Loaded CSI model (either CSIModel or CSIModelWithZoneMasking)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model architecture from config
    config = checkpoint.get('config')
    if config is None:
        if fallback_config is None:
            raise ValueError("Model checkpoint missing configuration information and no fallback config provided")
        logger.warning("Model checkpoint missing configuration information, using fallback config")
        config = fallback_config
    
    # Check state dict keys to detect model architecture
    state_dict_keys = set(checkpoint['model_state_dict'].keys())
    
    # Detect if this is a zone focus model by checking for zone-specific components
    has_zone_transforms = any('zone_feature_transforms' in key for key in state_dict_keys)
    has_direct_zone_classifiers = any(key.startswith('zone_classifiers.') for key in state_dict_keys)
    has_head_zone_classifiers = any('head.zone_classifiers' in key for key in state_dict_keys)
    
    if has_zone_transforms or has_direct_zone_classifiers or has_head_zone_classifiers:
        # This is a zone focus model - use build_model with zone focus enabled
        logger.info("Detected zone focus model architecture")
        model = build_model(config, use_zone_focus=True)
    else:
        # This is a standard model
        logger.info("Detected standard model architecture")
        model = CSIModel(
            backbone_arch=config.model_arch,
            n_classes_per_zone=5,
            pretrained=False
        )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def evaluate_model_on_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Evaluate model on a data loader.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        criterion: Loss function (optional)
        
    Returns:
        Tuple of (predictions, targets, probabilities, loss)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0.0
    
    # Check if model supports zone masking
    is_zone_masking_model = hasattr(model, 'ZONE_MAPPING')
    
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Evaluating"):
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
                outputs = model(images, file_ids)  # [batch_size, n_zones, n_classes]
            else:
                outputs = model(images)  # [batch_size, n_zones, n_classes]
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            
            # Convert logits to probabilities and class predictions
            probabilities = torch.softmax(outputs, dim=-1)  # [batch_size, n_zones, n_classes]
            pred_classes = torch.argmax(outputs, dim=-1)  # [batch_size, n_zones]
            
            # Store predictions, targets, and probabilities
            all_predictions.append(pred_classes.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())
    
    # Concatenate all predictions, targets, and probabilities
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)
    avg_loss = total_loss / len(data_loader) if criterion is not None else 0.0
    
    return predictions, targets, probabilities, avg_loss


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
    from ..evaluation.metrics.f1_score import compute_pytorch_f1_metrics
    from ..evaluation.metrics.classification import compute_accuracy
    
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
    from ..evaluation.metrics.f1_score import compute_pytorch_f1_metrics
    from ..evaluation.metrics.classification import compute_accuracy, compute_precision_recall_metrics
    
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


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    zone_names: Optional[List[str]] = None
) -> None:
    """
    Save predictions and targets to CSV file.
    
    Args:
        predictions: Predicted CSI class indices [num_samples, num_zones]
        targets: Ground truth CSI class indices [num_samples, num_zones]
        output_path: Path to save CSV file
        zone_names: Names of CSI zones
    """
    if zone_names is None:
        zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    
    # Create DataFrame
    data = {}
    for i, zone_name in enumerate(zone_names):
        data[f"{zone_name}_pred"] = predictions[:, i]
        data[f"{zone_name}_true"] = targets[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def evaluate_model(config, run_dir: Optional[Path] = None) -> None:
    """
    Main evaluation function.
    
    Args:
        config: Configuration object
        run_dir: Optional run directory to save outputs to. If None, uses config.evaluation_dir
    """
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_trained_model(config.model_path, device, fallback_config=config)
    
    # Create data loaders
    from ..data.dataloader import create_data_loaders
    _, val_loader, test_loader = create_data_loaders(config)
    
    # Zone names
    zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_predictions, val_targets, val_probabilities, val_loss = evaluate_model_on_loader(
        model, val_loader, device
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions, test_targets, test_probabilities, test_loss = evaluate_model_on_loader(
        model, test_loader, device
    )
    
    # Compute metrics
    val_zone_metrics = compute_zone_metrics(val_predictions, val_targets, zone_names)
    test_zone_metrics = compute_zone_metrics(test_predictions, test_targets, zone_names)
    
    val_overall_metrics = compute_overall_metrics(val_predictions, val_targets)
    test_overall_metrics = compute_overall_metrics(test_predictions, test_targets)
    
    # Compute confusion matrices
    val_confusion_matrices = compute_confusion_matrices_per_zone(val_predictions, val_targets)
    test_confusion_matrices = compute_confusion_matrices_per_zone(test_predictions, test_targets)
    
    # Create classification reports
    val_reports = create_classification_report_per_zone(val_predictions, val_targets)
    test_reports = create_classification_report_per_zone(test_predictions, test_targets)
    
    # Determine output directory
    if run_dir is not None:
        # Use run directory structure
        output_dir = run_dir / "evaluation"
        graphs_dir = run_dir / "graphs"
        roc_curves_dir = graphs_dir / "roc_curves"
        pr_curves_dir = graphs_dir / "pr_curves"
        confusion_matrices_dir = graphs_dir / "confusion_matrices"
        logger.info(f"Saving evaluation outputs to run directory: {output_dir}")
    else:
        # Use legacy evaluation directory
        output_dir = Path(config.evaluation_dir)
        graphs_dir = Path(config.graph_dir)
        roc_curves_dir = graphs_dir / "roc_curves"
        pr_curves_dir = graphs_dir / "pr_curves"
        confusion_matrices_dir = graphs_dir / "confusion_matrices"
        logger.info(f"Saving evaluation outputs to evaluation directory: {output_dir}")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_curves_dir.mkdir(parents=True, exist_ok=True)
    pr_curves_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrices_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config.ini and training history to run directory (for eval runs)
    if run_dir is not None:
        from src.utils.file_utils import copy_config_to_run_dir, copy_training_history_to_run_dir
        copy_config_to_run_dir(run_dir, config)
        copy_training_history_to_run_dir(run_dir, config)
    
    # Save predictions
    save_predictions(val_predictions, val_targets, output_dir / "val_predictions.csv", zone_names)
    save_predictions(test_predictions, test_targets, output_dir / "test_predictions.csv", zone_names)
    
    # Generate ROC curves
    logger.info("Generating ROC curves...")
    from .visualization.plots import create_roc_curves
    create_roc_curves(val_probabilities, val_targets, zone_names, class_names, 
                     str(roc_curves_dir), "validation")
    create_roc_curves(test_probabilities, test_targets, zone_names, class_names, 
                     str(roc_curves_dir), "test")
    
    # Generate Precision-Recall curves
    logger.info("Generating Precision-Recall curves...")
    from .visualization.plots import create_precision_recall_curves
    create_precision_recall_curves(val_probabilities, val_targets, zone_names, class_names, 
                                  str(pr_curves_dir), "validation")
    create_precision_recall_curves(test_probabilities, test_targets, zone_names, class_names, 
                                  str(pr_curves_dir), "test")
    
    # Save confusion matrix graphs
    from .visualization.confusion_matrix import save_confusion_matrix_graphs
    save_confusion_matrix_graphs(val_confusion_matrices, config, "evaluation", "validation", str(confusion_matrices_dir))
    save_confusion_matrix_graphs(test_confusion_matrices, config, "evaluation", "test", str(confusion_matrices_dir))
    
    # Save overall confusion matrix directly in graphs folder (not in subfolder)
    logger.info("Generating overall confusion matrix...")
    from .visualization.confusion_matrix import create_overall_confusion_matrix as create_overall_confusion_matrix_plot
    create_overall_confusion_matrix_plot(val_confusion_matrices, str(graphs_dir), "validation", config.model_arch)
    create_overall_confusion_matrix_plot(test_confusion_matrices, str(graphs_dir), "test", config.model_arch)
    
    # Generate AHF confusion matrix and save in graphs root
    logger.info("Generating AHF confusion matrix...")
    from .metrics.evaluation_metrics import create_ahf_confusion_matrix
    try:
        # Load CSV data for AHF analysis
        from ..data.preprocessing import load_csv_data, convert_nans_to_unknown
        csv_path = os.path.join(config.csv_dir, config.labels_csv)
        csv_data = load_csv_data(csv_path)
        csv_data = convert_nans_to_unknown(csv_data)
        
        # Create AHF confusion matrix for validation
        val_ahf_conf_matrix = create_ahf_confusion_matrix(val_predictions, val_targets, csv_data)
        if val_ahf_conf_matrix is not None:
            from .visualization.confusion_matrix import save_ahf_confusion_matrix
            save_ahf_confusion_matrix(val_ahf_conf_matrix, str(graphs_dir), "validation", config.model_arch)
        
        # Create AHF confusion matrix for test
        test_ahf_conf_matrix = create_ahf_confusion_matrix(test_predictions, test_targets, csv_data)
        if test_ahf_conf_matrix is not None:
            from .visualization.confusion_matrix import save_ahf_confusion_matrix
            save_ahf_confusion_matrix(test_ahf_conf_matrix, str(graphs_dir), "test", config.model_arch)
            
    except Exception as e:
        logger.warning(f"Could not generate AHF confusion matrix: {e}")
    
    # Generate summary dashboard (save in run directory root, not in graphs subfolder)
    logger.info("Generating summary dashboard...")
    from .visualization.plots import create_summary_dashboard
    from .metrics.evaluation_metrics import create_overall_confusion_matrix
    
    # Create overall confusion matrix for dashboard
    val_overall_conf_matrix = create_overall_confusion_matrix(val_predictions, val_targets)
    test_overall_conf_matrix = create_overall_confusion_matrix(test_predictions, test_targets)
    
    # For dashboard, we need training history - try to load it
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_precisions, val_precisions = [], []
    
    # Try to load training history from INI_DIR
    try:
        import json
        ini_dir_path = Path(config.ini_dir)
        training_history_path = ini_dir_path / "training_history.json"
        
        if training_history_path.exists():
            with open(training_history_path, 'r') as f:
                history = json.load(f)
            
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            train_accuracies = history.get('train_accuracies', [])
            val_accuracies = history.get('val_accuracies', [])
            train_f1_scores = history.get('train_f1_scores', [])
            val_f1_scores = history.get('val_f1_scores', [])
            train_precisions = history.get('train_precisions', [])
            val_precisions = history.get('val_precisions', [])
            logger.info("Loaded training history from INI_DIR for dashboard")
        else:
            logger.warning("Training history file not found in INI_DIR, using empty lists for dashboard")
    except Exception as e:
        logger.warning(f"Could not load training history from INI_DIR: {e}, using empty lists for dashboard")
    
    # Create validation dashboard (always create, even without training history)
    try:
        create_summary_dashboard(
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            train_f1_scores, val_f1_scores,
            train_precisions, val_precisions,
            val_probabilities, val_targets,
            zone_names, class_names,
            val_overall_conf_matrix,
            str(run_dir) if run_dir else str(graphs_dir),
            config.model_arch, "validation"
        )
        logger.info("Created validation summary dashboard")
    except Exception as e:
        logger.error(f"Failed to create validation dashboard: {e}")
    
    # Create test dashboard (always create, even without training history)
    try:
        create_summary_dashboard(
            train_losses, val_losses,
            train_accuracies, val_accuracies,
            train_f1_scores, val_f1_scores,
            train_precisions, val_precisions,
            test_probabilities, test_targets,
            zone_names, class_names,
            test_overall_conf_matrix,
            str(run_dir) if run_dir else str(graphs_dir),
            config.model_arch, "test"
        )
        logger.info("Created test summary dashboard")
    except Exception as e:
        logger.error(f"Failed to create test dashboard: {e}")
    
    # Log to wandb if enabled
    if hasattr(config, 'use_wandb') and config.use_wandb:
        log_to_wandb(
            val_overall_metrics, test_overall_metrics,
            val_confusion_matrices, test_confusion_matrices,
            val_reports, test_reports,
            config, config.model_path, "evaluation"
        )
    
    # Print summary
    logger.info("Evaluation completed!")
    logger.info(f"Validation - F1 Macro: {val_overall_metrics['f1_macro']:.4f}, Accuracy: {val_overall_metrics['accuracy']:.4f}")
    logger.info(f"Test - F1 Macro: {test_overall_metrics['f1_macro']:.4f}, Accuracy: {test_overall_metrics['accuracy']:.4f}")
    logger.info(f"All graphs saved to: {graphs_dir if run_dir else config.graph_dir}")
    if run_dir:
        logger.info(f"Summary dashboard saved to: {run_dir}")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 