"""
Confusion matrix visualization for CSI-Predictor evaluation.

This module contains confusion matrix visualization functionality extracted from the original src/utils.py file.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Optional
from pathlib import Path
from ...utils.logging import logger

def save_confusion_matrix_graphs(
    confusion_matrices: Dict[str, np.ndarray],
    config,
    run_name: str,
    split_name: str = "validation"
) -> None:
    """
    Save confusion matrix graphs for each zone.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices per zone
        config: Configuration object
        run_name: Name of the run
        split_name: Name of the data split
    """
    save_dir = Path(config.output_dir) / "confusion_matrices" / split_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    for zone_name in zone_names:
        if zone_name not in confusion_matrices:
            continue
            
        cm = confusion_matrices[zone_name]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title(f'Confusion Matrix - {zone_name} ({split_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save plot
        plot_path = save_dir / f"confusion_matrix_{zone_name}_{split_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix for {zone_name} to {plot_path}")


def create_confusion_matrix_grid(
    confusion_matrices: Dict[str, np.ndarray],
    save_dir: str,
    split_name: str = "validation",
    run_name: str = "model"
) -> None:
    """
    Create a grid of confusion matrices for all zones.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices per zone
        save_dir: Directory to save the grid
        split_name: Name of the data split
        run_name: Name of the run
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    zone_names = ["right_sup", "left_sup", "right_mid", "left_mid", "right_inf", "left_inf"]
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Confusion Matrices - {run_name} ({split_name})', fontsize=16)
    
    for idx, zone_name in enumerate(zone_names):
        if zone_name not in confusion_matrices:
            continue
            
        row = idx // 3
        col = idx % 3
        
        cm = confusion_matrices[zone_name]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[row, col])
        
        axes[row, col].set_title(f'{zone_name}')
        axes[row, col].set_xlabel('Predicted')
        axes[row, col].set_ylabel('Actual')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = save_path / f"confusion_matrix_grid_{split_name}_{run_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix grid to {plot_path}")


def create_overall_confusion_matrix(
    confusion_matrices: Dict[str, np.ndarray],
    save_dir: str,
    split_name: str = "validation",
    run_name: str = "model"
) -> None:
    """
    Create an overall confusion matrix by summing all zone confusion matrices.
    
    Args:
        confusion_matrices: Dictionary of confusion matrices per zone
        save_dir: Directory to save the overall matrix
        split_name: Name of the data split
        run_name: Name of the run
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    class_names = ["Normal", "Mild", "Moderate", "Severe", "Unknown"]
    
    # Sum all confusion matrices
    overall_cm = np.zeros((5, 5), dtype=int)
    for zone_name, cm in confusion_matrices.items():
        overall_cm += cm
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Overall Confusion Matrix - {run_name} ({split_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save plot
    plot_path = save_path / f"overall_confusion_matrix_{split_name}_{run_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved overall confusion matrix to {plot_path}")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 