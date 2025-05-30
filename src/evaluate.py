"""
Evaluation logic for CSI-Predictor.
Handles model evaluation, metrics computation, and result analysis.
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .data import create_data_loaders
from .models import CSIModel, build_model
from .config import cfg, get_config
from .metrics import compute_pytorch_f1_metrics, compute_accuracy


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


def create_evaluation_report(
    zone_metrics: Dict[str, Dict[str, float]],
    overall_metrics: Dict[str, float],
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[str] = None
) -> str:
    """
    Create evaluation report.
    
    Args:
        zone_metrics: Metrics per zone
        overall_metrics: Overall metrics
        predictions: Predictions array
        targets: Targets array
        output_path: Path to save report (optional)
        
    Returns:
        Report string
    """
    report_lines = []
    
    # Header
    report_lines.append("CSI-Predictor Evaluation Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("Overall Metrics:")
    report_lines.append("-" * 20)
    for metric, value in overall_metrics.items():
        if 'samples' in metric:
            report_lines.append(f"{metric}: {value}")
        else:
            report_lines.append(f"{metric}: {value:.4f}")
    report_lines.append("")
    
    # Zone-specific metrics
    report_lines.append("Zone-specific Metrics:")
    report_lines.append("-" * 25)
    
    # Create formatted table
    zone_names = list(zone_metrics.keys())
    metrics_names = ['f1_macro', 'f1_weighted', 'accuracy', 'valid_samples']
    
    # Header row
    header = f"{'Zone':<12}"
    for metric in metrics_names:
        if metric == 'valid_samples':
            header += f"{'Valid':<8}"
        else:
            header += f"{metric.replace('_', ' ').title():<12}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Data rows
    for zone_name in zone_names:
        row = f"{zone_name:<12}"
        for metric in metrics_names:
            value = zone_metrics[zone_name][metric]
            if metric == 'valid_samples':
                row += f"{value:<8}"
            else:
                row += f"{value:<12.4f}"
        report_lines.append(row)
    
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("Summary Statistics:")
    report_lines.append("-" * 20)
    report_lines.append(f"Number of samples: {len(predictions)}")
    report_lines.append(f"Number of zones: {predictions.shape[1]}")
    report_lines.append(f"Prediction range: [{predictions.min()}, {predictions.max()}]")
    report_lines.append(f"Target range: [{targets.min()}, {targets.max()}]")
    
    # Class distribution
    report_lines.append("")
    report_lines.append("Class Distribution:")
    report_lines.append("-" * 20)
    for class_idx in range(5):
        pred_count = (predictions == class_idx).sum()
        target_count = (targets == class_idx).sum()
        report_lines.append(f"Class {class_idx}: Pred={pred_count}, True={target_count}")
    
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
    Main evaluation function.
    
    Args:
        config: Configuration object
    """
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Find the latest model - look for timestamped models
    models_folder = Path(config.models_folder)
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
    zone_names = ["zone_1", "zone_2", "zone_3", "zone_4", "zone_5", "zone_6"]
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    val_predictions, val_targets, val_loss = evaluate_model_on_loader(model, val_loader, device, criterion)
    
    val_zone_metrics = compute_zone_metrics(val_predictions, val_targets, zone_names)
    val_overall_metrics = compute_overall_metrics(val_predictions, val_targets)
    val_overall_metrics['loss'] = val_loss
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions, test_targets, test_loss = evaluate_model_on_loader(model, test_loader, device, criterion)
    
    test_zone_metrics = compute_zone_metrics(test_predictions, test_targets, zone_names)
    test_overall_metrics = compute_overall_metrics(test_predictions, test_targets)
    test_overall_metrics['loss'] = test_loss
    
    # Create output directory
    output_dir = Path(config.models_folder) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and save reports
    val_report = create_evaluation_report(
        val_zone_metrics, val_overall_metrics, val_predictions, val_targets,
        output_dir / "validation_report.txt"
    )
    
    test_report = create_evaluation_report(
        test_zone_metrics, test_overall_metrics, test_predictions, test_targets,
        output_dir / "test_report.txt"
    )
    
    # Save predictions
    save_predictions(val_predictions, val_targets, output_dir / "validation_predictions.csv", zone_names)
    save_predictions(test_predictions, test_targets, output_dir / "test_predictions.csv", zone_names)
    
    # Print summary
    print("\nValidation Results:")
    print(val_report)
    print("\nTest Results:")
    print(test_report)
    
    logger.info("Evaluation completed!")


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