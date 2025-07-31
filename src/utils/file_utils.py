"""
File operation utilities for CSI-Predictor.

This module contains file operation functionality extracted from the original src/utils.py file.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import json
from src.utils.logging import logger

def create_dirs(*paths: str) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        *paths: Paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_training_history(
    train_losses: list,
    val_losses: list,
    train_accuracies: list,
    val_accuracies: list,
    train_precisions: list,
    val_precisions: list,
    train_f1_scores: list,
    val_f1_scores: list,
    save_path: str
) -> None:
    """
    Save training history to JSON file.
    
    Args:
        train_losses: Training losses
        val_losses: Validation losses
        train_accuracies: Training accuracies
        val_accuracies: Validation accuracies
        train_precisions: Training precisions
        val_precisions: Validation precisions
        train_f1_scores: Training F1 scores
        val_f1_scores: Validation F1 scores
        save_path: Path to save the history
    """
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_precisions': train_precisions,
        'val_precisions': val_precisions,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores
    }
    
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)


def load_training_history(history_path: str) -> tuple:
    """
    Load training history from JSON file.
    
    Args:
        history_path: Path to the history file
        
    Returns:
        Tuple of training history lists
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return (
        history['train_losses'],
        history['val_losses'],
        history['train_accuracies'],
        history['val_accuracies'],
        history['train_precisions'],
        history['val_precisions'],
        history['train_f1_scores'],
        history['val_f1_scores']
    )


def create_run_directory(config, run_type: str = "both") -> Path:
    """
    Create a timestamped run directory with proper structure.
    
    Args:
        config: Configuration object
        run_type: Type of run ("train", "eval", "both")
        
    Returns:
        Path to the created run directory
    """
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run directory name with model architecture and run type
    run_dir_name = f"{timestamp}_{config.model_arch}_{run_type}"
    run_dir_path = Path(config.runs_dir) / run_dir_name
    
    # Create main run directory
    run_dir_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir_path}")
    
    # Create subdirectories based on run type
    if run_type in ["both", "train"]:
        # Training-related directories
        graphs_dir = run_dir_path / "graphs"
        graphs_dir.mkdir(exist_ok=True)
        
        training_curves_dir = graphs_dir / "training_curves"
        training_curves_dir.mkdir(exist_ok=True)
        
        # Save training history JSON
        history_dir = run_dir_path / "training_history"
        history_dir.mkdir(exist_ok=True)
    
    if run_type in ["both", "eval"]:
        # Evaluation-related directories
        evaluation_dir = run_dir_path / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        
        confusion_matrices_dir = run_dir_path / "graphs" / "confusion_matrices"
        confusion_matrices_dir.mkdir(parents=True, exist_ok=True)
    
    # Always create graphs directory for any graphs that might be saved
    graphs_dir = run_dir_path / "graphs"
    graphs_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created run directory structure: {run_dir_path}")
    return run_dir_path

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 