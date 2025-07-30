"""
File operation utilities for CSI-Predictor.

This module contains file operation functionality extracted from the original src/utils.py file.
"""

from pathlib import Path
from typing import Dict, Any
import json

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

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 