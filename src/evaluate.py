"""
Evaluation logic for CSI-Predictor.
Handles model evaluation, metrics computation, result analysis, and WandB logging.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .data import create_data_loaders
from .models import CSIModel, build_model
from .config import cfg, get_config
from .metrics import compute_pytorch_f1_metrics, compute_accuracy, compute_confusion_matrix
from .utils import (
    EarlyStopping, MetricsTracker, calculate_class_weights,
    logger, make_run_name, make_model_name, seed_everything, log_config,
    create_roc_curves, create_precision_recall_curves,
    create_confusion_matrix_grid, create_roc_curves_grid, create_precision_recall_curves_grid,
    create_model_name_from_existing, create_overall_confusion_matrix, create_summary_dashboard,
    load_training_history
)


def load_trained_model(model_path: str, device: torch.device) -> CSIModel:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model architecture from config
    config = checkpoint.get('config')
    if config is None:
        raise ValueError("Model checkpoint missing configuration information")
    
    # Create model
    model = CSIModel(
        backbone_arch=config.model_arch,
        n_classes_per_zone=5,  # Classification with 5 classes
        pretrained=False  # Don't need pretrained weights when loading checkpoint
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


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
            # Convert to PyTorch tensors
            pred_tensor = torch.from_numpy(zone_pred)
            true_tensor = torch.from_numpy(zone_true)
            
            # Compute confusion matrix including all classes (0-4, including unknown)
            cm = compute_confusion_matrix(pred_tensor, true_tensor, num_classes=5)
            confusion_matrices[zone_name] = cm.numpy()
        else:
            # No samples for this zone
            confusion_matrices[zone_name] = np.zeros((5, 5))
    
    return confusion_matrices


def create_classification_report_per_zone(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Dict]:
    """
    Create detailed classification report for each zone using pure PyTorch.
    
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
        # Get all samples for this zone (including unknown class 4)
        zone_pred = predictions[:, i]
        zone_true = targets[:, i]
        
        if len(zone_pred) == 0:
            reports[zone_name] = {"note": "No samples for this zone"}
            continue
        
        # Convert to PyTorch tensors
        pred_tensor = torch.from_numpy(zone_pred)
        true_tensor = torch.from_numpy(zone_true)
        
        # Compute confusion matrix including all classes
        cm = compute_confusion_matrix(pred_tensor, true_tensor, num_classes=5).numpy()
        
        # Calculate per-class metrics for classes 0-3 (exclude unknown class from precision/recall)
        report = {}
        for class_idx in range(4):  # Only for classes 0-3 (Normal, Mild, Moderate, Severe)
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = (true_tensor == class_idx).sum().item()
            
            report[class_names[class_idx]] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "support": support
            }
        
        # Add unknown class statistics (no precision/recall, just support)
        unknown_support = (true_tensor == 4).sum().item()
        if unknown_support > 0:
            report["Unknown"] = {
                "precision": float('nan'),  # Not meaningful for unknown class
                "recall": float('nan'),     # Not meaningful for unknown class
                "f1-score": float('nan'),   # Not meaningful for unknown class
                "support": unknown_support
            }
        
        # Overall metrics for this zone (including unknown samples)
        accuracy = (pred_tensor == true_tensor).float().mean().item()
        total_support = len(zone_pred)
        
        # Calculate separate accuracy excluding unknown samples for medical evaluation
        valid_mask = true_tensor != 4
        if valid_mask.sum() > 0:
            valid_accuracy = (pred_tensor[valid_mask] == true_tensor[valid_mask]).float().mean().item()
            report["accuracy_valid_only"] = valid_accuracy
        
        report["accuracy"] = accuracy
        report["total_support"] = total_support
        report["unknown_samples"] = unknown_support
        reports[zone_name] = report
    
    return reports


def save_confusion_matrix_graphs(
    confusion_matrices: Dict[str, np.ndarray],
    config,
    run_name: str,
    split_name: str = "validation"
) -> None:
    """
    Save confusion matrices as PNG graphs in organized folders.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices per zone
        config: Configuration object with graph_dir
        run_name: Experiment run name
        split_name: Name of the data split (validation/test)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    # Extract model name from run name for consistent folder structure
    model_name = run_name.rsplit('_', 1)[0] if '_' in run_name else run_name
    
    # Create graphs directory structure with new organization
    graphs_dir = Path(config.graph_dir) / model_name / "confusion_matrices" / "individual_zones"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    # Zone names with corrected left/right orientation for chest X-rays
    # In chest X-rays viewed from front, patient's left lung appears on right side of image
    zone_display_names = {
        "right_sup": "Patient Right Superior",
        "left_sup": "Patient Left Superior", 
        "right_mid": "Patient Right Middle",
        "left_mid": "Patient Left Middle",
        "right_inf": "Patient Right Inferior",
        "left_inf": "Patient Left Inferior"
    }
    
    for zone_name, cm in confusion_matrices.items():
        if cm.sum() > 0:  # Only create graphs for zones with data
            plt.figure(figsize=(8, 6))
            
            # Convert to integers for proper formatting
            cm_int = cm.astype(int)
            
            # Create heatmap
            sns.heatmap(
                cm_int, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Count'}
            )
            
            display_name = zone_display_names.get(zone_name, zone_name.replace('_', ' ').title())
            plt.title(f'Confusion Matrix - {display_name}\n({split_name.title()} Set)')
            plt.xlabel('Predicted CSI Score')
            plt.ylabel('True CSI Score')
            plt.tight_layout()
            
            # Save graph in individual zones subfolder
            filename = f"{split_name}_{zone_name}_confusion_matrix.png"
            save_path = graphs_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved confusion matrix graph: {save_path}")


def log_to_wandb(
    val_metrics: Dict,
    test_metrics: Dict,
    val_confusion_matrices: Dict,
    test_confusion_matrices: Dict,
    val_reports: Dict,
    test_reports: Dict,
    config,
    model_path: str,
    eval_model_name: str
) -> None:
    """
    Log evaluation results to Weights & Biases without console confusion matrices.
    
    Args:
        val_metrics: Validation metrics
        test_metrics: Test metrics
        val_confusion_matrices: Validation confusion matrices per zone
        test_confusion_matrices: Test confusion matrices per zone
        val_reports: Validation classification reports per zone
        test_reports: Test classification reports per zone
        config: Configuration object
        model_path: Path to the evaluated model
        eval_model_name: Consistent model name for this evaluation run
    """
    try:
        import wandb
        
        # Initialize wandb run for evaluation
        wandb.init(
            project="csi-predictor-eval",
            name=eval_model_name,
            config={
                "model_arch": config.model_arch,
                "batch_size": config.batch_size,
                "model_path": model_path,
                "evaluation_mode": "full_evaluation"
            }
        )
        
        # Log overall metrics
        wandb.log({
            "val/loss": val_metrics.get("loss", 0),
            "val/f1_macro": val_metrics["overall_f1_macro"],
            "val/accuracy": val_metrics["overall_accuracy"],
            "test/loss": test_metrics.get("loss", 0),
            "test/f1_macro": test_metrics["overall_f1_macro"], 
            "test/accuracy": test_metrics["overall_accuracy"],
        })
        
        # Zone names with corrected orientation
        corrected_zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
        for zone in corrected_zone_names:
            if zone in val_reports:
                val_report = val_reports[zone]
                if "accuracy" in val_report:
                    wandb.log({f"val/{zone}/accuracy": val_report["accuracy"]})
                    
            if zone in test_reports:
                test_report = test_reports[zone]
                if "accuracy" in test_report:
                    wandb.log({f"test/{zone}/accuracy": test_report["accuracy"]})
        
        # Log confusion matrices as heatmaps (WandB only, not console)
        for split, cms in [("val", val_confusion_matrices), ("test", test_confusion_matrices)]:
            for zone, cm in cms.items():
                if cm.sum() > 0:  # Only log if there are samples
                    # Create wandb Table for confusion matrix
                    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
                    wandb.log({
                        f"{split}/{zone}/confusion_matrix": wandb.plots.HeatMap(
                            x_labels=class_names,
                            y_labels=class_names,
                            matrix_values=cm.tolist(),
                            show_text=True
                        )
                    })
        
        # Log detailed classification reports as tables
        for split, reports in [("val", val_reports), ("test", test_reports)]:
            for zone, report in reports.items():
                if "accuracy" in report:  # Valid report
                    # Create table for per-class metrics
                    table_data = []
                    class_names = ["Normal", "Mild", "Moderate", "Severe"]
                    
                    for class_name in class_names:
                        if class_name in report:
                            metrics = report[class_name]
                            table_data.append([
                                class_name,
                                f"{metrics['precision']:.3f}",
                                f"{metrics['recall']:.3f}",
                                f"{metrics['f1-score']:.3f}",
                                metrics['support']
                            ])
                    
                    if table_data:
                        table = wandb.Table(
                            columns=["Class", "Precision", "Recall", "F1-Score", "Support"],
                            data=table_data
                        )
                        wandb.log({f"{split}/{zone}/classification_report": table})
        
        logger.info("Successfully logged evaluation results to WandB")
        wandb.finish()
        
    except Exception as e:
        logger.warning(f"Could not log to WandB: {e}")


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
    Compute overall metrics across all zones (classification version).
    
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
    f1_metrics = compute_pytorch_f1_metrics(logits, target_tensor, ignore_index=4)
    accuracy_metrics = compute_accuracy(logits, target_tensor, ignore_index=4)
    
    # Import precision/recall metrics
    from .metrics import compute_precision_recall_metrics
    pr_metrics = compute_precision_recall_metrics(logits, target_tensor, ignore_index=4)
    
    return {
        'overall_f1_macro': f1_metrics['f1_overall'],
        'overall_f1_weighted': f1_metrics['f1_overall'],  # Use same as macro for simplicity
        'overall_accuracy': accuracy_metrics['acc_overall'],
        'overall_precision_macro': pr_metrics['precision_overall'],
        'overall_recall_macro': pr_metrics['recall_overall'],
        'total_valid_samples': int((targets != 4).sum())
    }


def create_evaluation_report(
    zone_metrics: Dict[str, Dict[str, float]],
    overall_metrics: Dict[str, float],
    predictions: np.ndarray,
    targets: np.ndarray,
    confusion_matrices: Dict[str, np.ndarray],
    classification_reports: Dict[str, Dict],
    output_path: Optional[str] = None,
    include_confusion_matrices: bool = True
) -> str:
    """
    Create comprehensive evaluation report with optional confusion matrices.
    
    Args:
        zone_metrics: Metrics per zone
        overall_metrics: Overall metrics
        predictions: Predictions array
        targets: Targets array
        confusion_matrices: Confusion matrices per zone
        classification_reports: Classification reports per zone
        output_path: Path to save report (optional)
        include_confusion_matrices: Whether to include confusion matrices in text output
        
    Returns:
        Report string
    """
    report_lines = []
    
    # Header
    report_lines.append("CSI-Predictor Comprehensive Evaluation Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("📊 Overall Metrics:")
    report_lines.append("-" * 20)
    for metric, value in overall_metrics.items():
        if 'samples' in metric:
            report_lines.append(f"{metric}: {value}")
        else:
            report_lines.append(f"{metric}: {value:.4f}")
    report_lines.append("")
    
    # Zone-specific metrics with corrected orientation labels
    report_lines.append("🏥 Zone-specific Metrics (Corrected Orientation):")
    report_lines.append("-" * 50)
    
    # Zone display names for chest X-rays (viewed from front)
    zone_display_mapping = {
        "right_sup": "Patient Right Superior",
        "left_sup": "Patient Left Superior", 
        "right_mid": "Patient Right Middle",
        "left_mid": "Patient Left Middle",
        "right_inf": "Patient Right Inferior",
        "left_inf": "Patient Left Inferior"
    }
    
    # Create formatted table
    zone_names = list(zone_metrics.keys())
    metrics_names = ['f1_macro', 'f1_weighted', 'accuracy', 'valid_samples']
    
    # Header row
    header = f"{'Zone':<25}"
    for metric in metrics_names:
        if metric == 'valid_samples':
            header += f"{'Valid':<8}"
        else:
            header += f"{metric.replace('_', ' ').title():<12}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Data rows with corrected orientation
    for zone_name in zone_names:
        display_name = zone_display_mapping.get(zone_name, zone_name.replace('_', ' ').title())
        row = f"{display_name:<25}"
        for metric in metrics_names:
            value = zone_metrics[zone_name][metric]
            if metric == 'valid_samples':
                row += f"{value:<8}"
            else:
                row += f"{value:<12.4f}"
        report_lines.append(row)
    
    report_lines.append("")
    
    # Confusion matrices per zone (optional)
    if include_confusion_matrices:
        report_lines.append("🔢 Confusion Matrices per Zone:")
        report_lines.append("-" * 35)
        class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
        
        for zone_name, cm in confusion_matrices.items():
            if cm.sum() > 0:  # Only show if there are samples
                display_name = zone_display_mapping.get(zone_name, zone_name.replace('_', ' ').title())
                report_lines.append(f"\n{display_name}:")
                report_lines.append("    " + "".join([f"{name:<8}" for name in class_names]))
                for i, class_name in enumerate(class_names):
                    row_str = f"{class_name:<4}"
                    for j in range(5):
                        row_str += f"{int(cm[i, j]):<8}"
                    report_lines.append(row_str)
        
        report_lines.append("")
    
    # Classification reports per zone
    report_lines.append("📋 Detailed Classification Reports:")
    report_lines.append("-" * 40)
    
    for zone_name, report in classification_reports.items():
        if "accuracy" in report:  # Valid report
            display_name = zone_display_mapping.get(zone_name, zone_name.replace('_', ' ').title())
            report_lines.append(f"\n{display_name}:")
            report_lines.append(f"Overall Accuracy: {report['accuracy']:.4f}")
            report_lines.append(f"Total Support: {report['total_support']}")
            
            # Per-class metrics
            report_lines.append(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            report_lines.append("-" * 50)
            
            for class_name in ["Normal", "Mild", "Moderate", "Severe"]:
                if class_name in report:
                    metrics = report[class_name]
                    report_lines.append(
                        f"{class_name:<10} "
                        f"{metrics['precision']:<10.3f} "
                        f"{metrics['recall']:<10.3f} "
                        f"{metrics['f1-score']:<10.3f} "
                        f"{metrics['support']:<10}"
                    )
    
    # Summary statistics
    report_lines.append("\n📈 Summary Statistics:")
    report_lines.append("-" * 20)
    report_lines.append(f"Number of samples: {len(predictions)}")
    report_lines.append(f"Number of zones: {predictions.shape[1]}")
    report_lines.append(f"Prediction range: [{predictions.min()}, {predictions.max()}]")
    report_lines.append(f"Target range: [{targets.min()}, {targets.max()}]")
    
    # Class distribution
    report_lines.append("")
    report_lines.append("📊 Class Distribution:")
    report_lines.append("-" * 20)
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    for class_idx in range(5):
        pred_count = (predictions == class_idx).sum()
        target_count = (targets == class_idx).sum()
        report_lines.append(f"Class {class_idx} ({class_names[class_idx]}): Pred={pred_count}, True={target_count}")
    
    report = "\n".join(report_lines)
    
    # Save report if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation report saved to {output_path}")
    
    return report


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str,
    zone_names: Optional[List[str]] = None
) -> None:
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Predicted CSI class indices
        targets: Ground truth CSI class indices
        output_path: Path to save CSV
        zone_names: Names of zones (optional)
    """
    if zone_names is None:
        zone_names = [f"zone_{i+1}" for i in range(predictions.shape[1])]
    
    # Create DataFrame
    data = {}
    
    # Add predictions
    for i, zone_name in enumerate(zone_names):
        data[f"pred_{zone_name}"] = predictions[:, i]
        data[f"true_{zone_name}"] = targets[:, i]
        data[f"correct_{zone_name}"] = (predictions[:, i] == targets[:, i]).astype(int)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def evaluate_model(config) -> None:
    """
    Evaluate a trained CSI-Predictor model on validation and test sets.
    
    Args:
        config: Configuration object
    """
    # Set seed for reproducibility
    seed_everything(42)
    
    # Log configuration
    log_config(config)
    
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Find the latest model - look for timestamped models
    models_folder = Path(config.models_dir)
    if not models_folder.exists():
        raise FileNotFoundError(f"Models folder not found: {models_folder}")
    
    # Look for model files
    model_files = list(models_folder.glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_folder}")
    
    # Use the most recent model file
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Using model: {latest_model}")
    
    # Extract model name from the file and create evaluation run name
    eval_model_name = create_model_name_from_existing(str(latest_model))
    eval_run_name = make_run_name(eval_model_name, task_tag="Eval")
    logger.info(f"Model name: {eval_model_name}")
    logger.info(f"Evaluation run name: {eval_run_name}")

    model = load_trained_model(str(latest_model), device)
    
    # Create data loaders
    _, val_loader, test_loader = create_data_loaders(config)
    
    # Import the loss function from train.py
    from .train import WeightedCSILoss
    criterion = WeightedCSILoss(unknown_weight=0.3)
    criterion = criterion.to(device)  # Move criterion to same device as model
    
    # Zone names
    zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_predictions, val_targets, val_probabilities, val_loss = evaluate_model_on_loader(model, val_loader, device, criterion)
    
    # Debug: Check for unknown samples in validation set
    val_unknown_count = (val_targets == 4).sum()
    logger.info(f"Validation set has {val_unknown_count} unknown labels (class 4) out of {val_targets.size} total labels")
    for i, zone_name in enumerate(zone_names):
        zone_unknown = (val_targets[:, i] == 4).sum()
        logger.info(f"  {zone_name}: {zone_unknown} unknown labels")
    
    val_zone_metrics = compute_zone_metrics(val_predictions, val_targets, zone_names)
    val_overall_metrics = compute_overall_metrics(val_predictions, val_targets)
    val_overall_metrics['loss'] = val_loss
    
    # Compute confusion matrices and classification reports for validation
    val_confusion_matrices = compute_confusion_matrices_per_zone(val_predictions, val_targets)
    val_classification_reports = create_classification_report_per_zone(val_predictions, val_targets)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions, test_targets, test_probabilities, test_loss = evaluate_model_on_loader(model, test_loader, device, criterion)
    
    # Debug: Check for unknown samples in test set
    test_unknown_count = (test_targets == 4).sum()
    logger.info(f"Test set has {test_unknown_count} unknown labels (class 4) out of {test_targets.size} total labels")
    for i, zone_name in enumerate(zone_names):
        zone_unknown = (test_targets[:, i] == 4).sum()
        logger.info(f"  {zone_name}: {zone_unknown} unknown labels")
    
    test_zone_metrics = compute_zone_metrics(test_predictions, test_targets, zone_names)
    test_overall_metrics = compute_overall_metrics(test_predictions, test_targets)
    test_overall_metrics['loss'] = test_loss
    
    # Compute confusion matrices and classification reports for test
    test_confusion_matrices = compute_confusion_matrices_per_zone(test_predictions, test_targets)
    test_classification_reports = create_classification_report_per_zone(test_predictions, test_targets)
    
    # Create output directory
    output_dir = Path(config.models_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save comprehensive reports
    val_report = create_evaluation_report(
        val_zone_metrics, val_overall_metrics, val_predictions, val_targets,
        val_confusion_matrices, val_classification_reports,
        output_dir / "validation_comprehensive_report.txt", include_confusion_matrices=False
    )
    
    test_report = create_evaluation_report(
        test_zone_metrics, test_overall_metrics, test_predictions, test_targets,
        test_confusion_matrices, test_classification_reports,
        output_dir / "test_comprehensive_report.txt", include_confusion_matrices=False
    )
    
    # Save predictions
    save_predictions(val_predictions, val_targets, output_dir / "validation_predictions.csv", zone_names)
    save_predictions(test_predictions, test_targets, output_dir / "test_predictions.csv", zone_names)
    
    # Save confusion matrix graphs
    save_confusion_matrix_graphs(val_confusion_matrices, config, eval_run_name, "validation")
    save_confusion_matrix_graphs(test_confusion_matrices, config, eval_run_name, "test")
    
    # Create and save ROC curves
    logger.info("Creating ROC curves...")
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    graphs_dir = Path(config.graph_dir) / eval_model_name
    
    val_roc_metrics = create_roc_curves(
        val_probabilities, val_targets, zone_names, class_names,
        str(graphs_dir), "validation", ignore_class=4
    )
    
    test_roc_metrics = create_roc_curves(
        test_probabilities, test_targets, zone_names, class_names,
        str(graphs_dir), "test", ignore_class=4
    )
    
    # Create and save Precision-Recall curves
    logger.info("Creating Precision-Recall curves...")
    
    val_pr_metrics = create_precision_recall_curves(
        val_probabilities, val_targets, zone_names, class_names,
        str(graphs_dir), "validation", ignore_class=4
    )
    
    test_pr_metrics = create_precision_recall_curves(
        test_probabilities, test_targets, zone_names, class_names,
        str(graphs_dir), "test", ignore_class=4
    )
    
    # Create and save grid layouts and overall visualizations
    logger.info("Creating comprehensive visualizations...")
    
    # Confusion matrix grids and overall plots
    create_confusion_matrix_grid(
        val_confusion_matrices, str(graphs_dir), "validation",
        eval_run_name
    )
    
    create_confusion_matrix_grid(
        test_confusion_matrices, str(graphs_dir), "test",
        eval_run_name
    )
    
    # Overall confusion matrices
    create_overall_confusion_matrix(
        val_confusion_matrices, str(graphs_dir), "validation", eval_run_name
    )
    
    create_overall_confusion_matrix(
        test_confusion_matrices, str(graphs_dir), "test", eval_run_name
    )
    
    # ROC curves grids
    create_roc_curves_grid(
        val_probabilities, val_targets, zone_names, class_names,
        str(graphs_dir), "validation", ignore_class=4
    )
    
    create_roc_curves_grid(
        test_probabilities, test_targets, zone_names, class_names,
        str(graphs_dir), "test", ignore_class=4
    )
    
    # Precision-Recall curves grids
    create_precision_recall_curves_grid(
        val_probabilities, val_targets, zone_names, class_names,
        str(graphs_dir), "validation", ignore_class=4
    )
    
    create_precision_recall_curves_grid(
        test_probabilities, test_targets, zone_names, class_names,
        str(graphs_dir), "test", ignore_class=4
    )
    
    # Create comprehensive summary dashboards
    logger.info("Creating evaluation summary dashboards...")
    
    # Load training history to display training curves
    history_path = graphs_dir / "training_history.json"
    (train_losses, val_losses, train_accuracies, val_accuracies,
     train_precisions, val_precisions, train_f1_scores, val_f1_scores, epochs_list) = load_training_history(str(history_path))
    
    # Get overall confusion matrix for validation
    val_overall_cm = sum([cm for cm in val_confusion_matrices.values() if cm.sum() > 0])
    test_overall_cm = sum([cm for cm in test_confusion_matrices.values() if cm.sum() > 0])
    
    # Create validation summary dashboard
    create_summary_dashboard(
        train_accuracies, val_accuracies,  # Load actual training curves
        train_losses, val_losses,  # Load actual training losses
        train_precisions, val_precisions,  # Load actual training precisions
        train_f1_scores, val_f1_scores,  # Load actual training F1 scores
        val_overall_cm, val_roc_metrics,
        str(graphs_dir), f"{eval_run_name}_validation", epochs_list,
        evaluation_metrics=val_overall_metrics
    )
    
    # Create test summary dashboard
    create_summary_dashboard(
        train_accuracies, val_accuracies,  # Load actual training curves
        train_losses, val_losses,  # Load actual training losses
        train_precisions, val_precisions,  # Load actual training precisions
        train_f1_scores, val_f1_scores,  # Load actual training F1 scores
        test_overall_cm, test_roc_metrics,
        str(graphs_dir), f"{eval_run_name}_test", epochs_list,
        evaluation_metrics=test_overall_metrics
    )
    
    # Log to WandB
    logger.info("Logging results to Weights & Biases...")
    log_to_wandb(
        val_overall_metrics, test_overall_metrics,
        val_confusion_matrices, test_confusion_matrices,
        val_classification_reports, test_classification_reports,
        config, str(latest_model), eval_run_name
    )
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 VALIDATION RESULTS:")
    print("="*80)
    print(val_report)
    
    print("\n" + "="*80)
    print("🏆 TEST RESULTS:")
    print("="*80)
    print(test_report)
    
    logger.info("Comprehensive evaluation completed!")


def main():
    """Main function for CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate CSI-Predictor model")
    parser.add_argument("--ini", default="config.ini", help="Path to config.ini file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = get_config(ini_path=args.ini)
    
    # Start evaluation
    evaluate_model(config)


if __name__ == "__main__":
    main() 