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
from .utils import logger, make_run_name, seed_everything, log_config


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
        # Get valid (non-unknown) samples for this zone
        zone_mask = targets[:, i] != 4
        if zone_mask.sum() > 0:
            zone_pred = predictions[:, i][zone_mask]
            zone_true = targets[:, i][zone_mask]
            
            # Convert to PyTorch tensors
            pred_tensor = torch.from_numpy(zone_pred)
            true_tensor = torch.from_numpy(zone_true)
            
            # Compute confusion matrix
            cm = compute_confusion_matrix(pred_tensor, true_tensor, num_classes=5)
            confusion_matrices[zone_name] = cm.numpy()
        else:
            # No valid samples for this zone
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
        # Get valid (non-unknown) samples for this zone
        zone_mask = targets[:, i] != 4
        zone_pred = predictions[:, i][zone_mask]
        zone_true = targets[:, i][zone_mask]
        
        if len(zone_pred) == 0:
            reports[zone_name] = {"note": "No valid samples for this zone"}
            continue
        
        # Convert to PyTorch tensors
        pred_tensor = torch.from_numpy(zone_pred)
        true_tensor = torch.from_numpy(zone_true)
        
        # Compute confusion matrix
        cm = compute_confusion_matrix(pred_tensor, true_tensor, num_classes=5).numpy()
        
        # Calculate per-class metrics
        report = {}
        for class_idx in range(5):
            if class_idx == 4:  # Skip unknown class for per-class metrics
                continue
                
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
        
        # Overall metrics for this zone
        accuracy = (pred_tensor == true_tensor).float().mean().item()
        total_support = len(zone_pred)
        
        report["accuracy"] = accuracy
        report["total_support"] = total_support
        reports[zone_name] = report
    
    return reports


def log_to_wandb(
    val_metrics: Dict,
    test_metrics: Dict,
    val_confusion_matrices: Dict,
    test_confusion_matrices: Dict,
    val_reports: Dict,
    test_reports: Dict,
    config,
    model_path: str
) -> None:
    """
    Log evaluation results to Weights & Biases.
    
    Args:
        val_metrics: Validation metrics
        test_metrics: Test metrics
        val_confusion_matrices: Validation confusion matrices per zone
        test_confusion_matrices: Test confusion matrices per zone
        val_reports: Validation classification reports per zone
        test_reports: Test classification reports per zone
        config: Configuration object
        model_path: Path to the evaluated model
    """
    try:
        import wandb
        
        # Initialize wandb run for evaluation
        run_name = f"eval_{make_run_name(config)}"
        wandb.init(
            project="csi-predictor-eval",
            name=run_name,
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
        
        # Log per-zone metrics
        zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
        for zone in zone_names:
            if zone in val_reports:
                val_report = val_reports[zone]
                if "accuracy" in val_report:
                    wandb.log({f"val/{zone}/accuracy": val_report["accuracy"]})
                    
            if zone in test_reports:
                test_report = test_reports[zone]
                if "accuracy" in test_report:
                    wandb.log({f"test/{zone}/accuracy": test_report["accuracy"]})
        
        # Log confusion matrices as heatmaps
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate model on a data loader.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to use
        criterion: Loss function (optional)
        
    Returns:
        Tuple of (predictions, targets, loss)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)  # [batch_size, n_zones, n_classes]
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            
            # Convert logits to class predictions
            pred_classes = torch.argmax(outputs, dim=-1)  # [batch_size, n_zones]
            
            # Store predictions and targets
            all_predictions.append(pred_classes.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(data_loader) if criterion is not None else 0.0
    
    return predictions, targets, avg_loss


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
    
    # Reorganize into per-zone format
    zone_metrics = {}
    for i, zone_name in enumerate(zone_names):
        zone_metrics[zone_name] = {
            'f1_macro': f1_metrics[f'f1_{zone_name}'],
            'f1_weighted': f1_metrics[f'f1_{zone_name}'],  # Use same as macro for simplicity
            'accuracy': accuracy_metrics[f'acc_{zone_name}'],
            'valid_samples': int((targets[:, i] != 4).sum())
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
    
    return {
        'overall_f1_macro': f1_metrics['f1_overall'],
        'overall_f1_weighted': f1_metrics['f1_overall'],  # Use same as macro for simplicity
        'overall_accuracy': accuracy_metrics['acc_overall'],
        'total_valid_samples': int((targets != 4).sum())
    }


def create_evaluation_report(
    zone_metrics: Dict[str, Dict[str, float]],
    overall_metrics: Dict[str, float],
    predictions: np.ndarray,
    targets: np.ndarray,
    confusion_matrices: Dict[str, np.ndarray],
    classification_reports: Dict[str, Dict],
    output_path: Optional[str] = None
) -> str:
    """
    Create comprehensive evaluation report with confusion matrices and classification reports.
    
    Args:
        zone_metrics: Metrics per zone
        overall_metrics: Overall metrics
        predictions: Predictions array
        targets: Targets array
        confusion_matrices: Confusion matrices per zone
        classification_reports: Classification reports per zone
        output_path: Path to save report (optional)
        
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
    
    # Zone-specific metrics
    report_lines.append("🏥 Zone-specific Metrics:")
    report_lines.append("-" * 25)
    
    # Create formatted table
    zone_names = list(zone_metrics.keys())
    metrics_names = ['f1_macro', 'f1_weighted', 'accuracy', 'valid_samples']
    
    # Header row
    header = f"{'Zone':<15}"
    for metric in metrics_names:
        if metric == 'valid_samples':
            header += f"{'Valid':<8}"
        else:
            header += f"{metric.replace('_', ' ').title():<12}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Data rows
    for zone_name in zone_names:
        row = f"{zone_name:<15}"
        for metric in metrics_names:
            value = zone_metrics[zone_name][metric]
            if metric == 'valid_samples':
                row += f"{value:<8}"
            else:
                row += f"{value:<12.4f}"
        report_lines.append(row)
    
    report_lines.append("")
    
    # Confusion matrices per zone
    report_lines.append("🔢 Confusion Matrices per Zone:")
    report_lines.append("-" * 35)
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    for zone_name, cm in confusion_matrices.items():
        if cm.sum() > 0:  # Only show if there are samples
            report_lines.append(f"\n{zone_name.replace('_', ' ').title()} Zone:")
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
            report_lines.append(f"\n{zone_name.replace('_', ' ').title()} Zone:")
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
    Main evaluation function with comprehensive reporting and WandB logging.
    
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
    
    model = load_trained_model(str(latest_model), device)
    
    # Create data loaders
    _, val_loader, test_loader = create_data_loaders(config)
    
    # Import the loss function from train.py
    from .train import MaskedCrossEntropyLoss
    criterion = MaskedCrossEntropyLoss(ignore_index=4)
    
    # Zone names
    zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_predictions, val_targets, val_loss = evaluate_model_on_loader(model, val_loader, device, criterion)
    
    val_zone_metrics = compute_zone_metrics(val_predictions, val_targets, zone_names)
    val_overall_metrics = compute_overall_metrics(val_predictions, val_targets)
    val_overall_metrics['loss'] = val_loss
    
    # Compute confusion matrices and classification reports for validation
    val_confusion_matrices = compute_confusion_matrices_per_zone(val_predictions, val_targets)
    val_classification_reports = create_classification_report_per_zone(val_predictions, val_targets)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions, test_targets, test_loss = evaluate_model_on_loader(model, test_loader, device, criterion)
    
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
        output_dir / "validation_comprehensive_report.txt"
    )
    
    test_report = create_evaluation_report(
        test_zone_metrics, test_overall_metrics, test_predictions, test_targets,
        test_confusion_matrices, test_classification_reports,
        output_dir / "test_comprehensive_report.txt"
    )
    
    # Save predictions
    save_predictions(val_predictions, val_targets, output_dir / "validation_predictions.csv", zone_names)
    save_predictions(test_predictions, test_targets, output_dir / "test_predictions.csv", zone_names)
    
    # Log to WandB
    logger.info("Logging results to Weights & Biases...")
    log_to_wandb(
        val_overall_metrics, test_overall_metrics,
        val_confusion_matrices, test_confusion_matrices,
        val_classification_reports, test_classification_reports,
        config, str(latest_model)
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