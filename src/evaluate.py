"""
Evaluation logic for CSI-Predictor.
Handles model evaluation, metrics computation, and result analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from .data import create_data_loaders
from .train import CSIModel
from .config import Config


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
        num_zones=6,
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
    Compute metrics for each CSI zone.
    
    Args:
        predictions: Predicted CSI scores [num_samples, num_zones]
        targets: Ground truth CSI scores [num_samples, num_zones]
        zone_names: Names of CSI zones
        
    Returns:
        Dictionary of metrics per zone
    """
    zone_metrics = {}
    
    for i, zone_name in enumerate(zone_names):
        zone_pred = predictions[:, i]
        zone_target = targets[:, i]
        
        zone_metrics[zone_name] = {
            'mse': mean_squared_error(zone_target, zone_pred),
            'rmse': np.sqrt(mean_squared_error(zone_target, zone_pred)),
            'mae': mean_absolute_error(zone_target, zone_pred),
            'r2': r2_score(zone_target, zone_pred)
        }
    
    return zone_metrics


def compute_overall_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute overall metrics across all zones.
    
    Args:
        predictions: Predicted CSI scores [num_samples, num_zones]
        targets: Ground truth CSI scores [num_samples, num_zones]
        
    Returns:
        Dictionary of overall metrics
    """
    # Flatten predictions and targets
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    return {
        'overall_mse': mean_squared_error(target_flat, pred_flat),
        'overall_rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
        'overall_mae': mean_absolute_error(target_flat, pred_flat),
        'overall_r2': r2_score(target_flat, pred_flat)
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
            outputs = model(images)
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item()
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
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
        report_lines.append(f"{metric}: {value:.4f}")
    report_lines.append("")
    
    # Zone-specific metrics
    report_lines.append("Zone-specific Metrics:")
    report_lines.append("-" * 25)
    
    # Create formatted table
    zone_names = list(zone_metrics.keys())
    metrics_names = ['mse', 'rmse', 'mae', 'r2']
    
    # Header row
    header = f"{'Zone':<12}"
    for metric in metrics_names:
        header += f"{metric.upper():<8}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    
    # Data rows
    for zone_name in zone_names:
        row = f"{zone_name:<12}"
        for metric in metrics_names:
            value = zone_metrics[zone_name][metric]
            row += f"{value:<8.4f}"
        report_lines.append(row)
    
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("Summary Statistics:")
    report_lines.append("-" * 20)
    report_lines.append(f"Number of samples: {len(predictions)}")
    report_lines.append(f"Number of zones: {predictions.shape[1]}")
    report_lines.append(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    report_lines.append(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
    
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
        predictions: Predicted CSI scores
        targets: Ground truth CSI scores
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
        data[f"error_{zone_name}"] = predictions[:, i] - targets[:, i]
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def evaluate_model(config: Config) -> None:
    """
    Main evaluation function.
    
    Args:
        config: Configuration object
    """
    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model - use best_model as default
    model_path = config.get_model_path("best_model")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = load_trained_model(model_path, device)
    
    # Create data loaders
    _, val_loader, test_loader = create_data_loaders(config)
    
    # Create loss function
    criterion = nn.MSELoss()
    
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