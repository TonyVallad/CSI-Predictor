"""
File operation utilities for CSI-Predictor.

This module contains file operation functionality extracted from the original src/utils.py file.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
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
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    train_precisions: List[float],
    val_precisions: List[float],
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    save_path: str
) -> None:
    """
    Save training history to JSON file.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accuracies: Training accuracies per epoch
        val_accuracies: Validation accuracies per epoch
        train_precisions: Training precisions per epoch
        val_precisions: Validation precisions per epoch
        train_f1_scores: Training F1 scores per epoch
        val_f1_scores: Validation F1 scores per epoch
        save_path: Path to save the JSON file
    """
    import json
    from pathlib import Path
    
    # Create training history dictionary
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_precisions': train_precisions,
        'val_precisions': val_precisions,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'epochs': list(range(1, len(train_losses) + 1))
    }
    
    # Save to specified path
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training history saved to: {save_path}")


def save_training_history_to_ini_dir(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    train_precisions: List[float],
    val_precisions: List[float],
    train_f1_scores: List[float],
    val_f1_scores: List[float],
    config
) -> str:
    """
    Save training history to INI_DIR folder (like config.ini).
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_accuracies: Training accuracies per epoch
        val_accuracies: Validation accuracies per epoch
        train_precisions: Training precisions per epoch
        val_precisions: Validation precisions per epoch
        train_f1_scores: Training F1 scores per epoch
        val_f1_scores: Validation F1 scores per epoch
        config: Configuration object
        
    Returns:
        Path to the saved training history file
    """
    # Save to INI_DIR with simple filename
    ini_dir_path = Path(config.ini_dir)
    ini_dir_path.mkdir(parents=True, exist_ok=True)
    
    save_path = ini_dir_path / "training_history.json"
    
    save_training_history(
        train_losses, val_losses,
        train_accuracies, val_accuracies,
        train_precisions, val_precisions,
        train_f1_scores, val_f1_scores,
        str(save_path)
    )
    
    return str(save_path)


def copy_training_history_to_run_dir(run_dir: Path, config) -> None:
    """
    Copy the training history file to the run directory.
    
    Args:
        run_dir: Run directory to copy training history to
        config: Configuration object
    """
    # Find the training history file in INI_DIR
    ini_dir_path = Path(config.ini_dir)
    training_history_path = ini_dir_path / "training_history.json"
    
    if not training_history_path.exists():
        logger.warning("Training history file not found in INI_DIR")
        return
    
    # Copy to run directory
    run_history_path = run_dir / "training_history.json"
    
    try:
        import shutil
        shutil.copy2(training_history_path, run_history_path)
        logger.info(f"Copied training history to run directory: {run_history_path}")
    except Exception as e:
        logger.error(f"Failed to copy training history: {e}")


def copy_config_to_run_dir(run_dir: Path, config) -> None:
    """
    Copy config.ini to the run directory.
    
    Args:
        run_dir: Run directory to copy config to
        config: Configuration object
    """
    # Find the config.ini file that was used
    original_config_path = None
    
    # Try to find the config.ini file that was actually used
    possible_paths = [
        "config/config.ini",
        "../config/config.ini", 
        "../../config/config.ini",
        "src/../config/config.ini",
    ]
    
    # Also check if we can determine the path from the config loader
    try:
        from src.config.config_loader import ConfigLoader
        temp_loader = ConfigLoader(".env", "dummy.ini")
        env_vars = temp_loader.load_env_vars()
        ini_dir = env_vars.get("INI_DIR", "")
        if ini_dir:
            possible_paths.insert(0, os.path.join(ini_dir, "config.ini"))
        else:
            data_dir = env_vars.get("DATA_DIR", "./data")
            possible_paths.insert(0, os.path.join(data_dir, "config", "config.ini"))
    except Exception:
        pass
    
    # Find the first existing config.ini file
    for path in possible_paths:
        if os.path.exists(path):
            original_config_path = path
            break
    
    if original_config_path and os.path.exists(original_config_path):
        # Copy config.ini to run directory
        run_config_path = run_dir / "config.ini"
        try:
            import shutil
            shutil.copy2(original_config_path, run_config_path)
            logger.info(f"Copied config.ini to run directory: {run_config_path}")
        except Exception as e:
            logger.error(f"Failed to copy config.ini: {e}")
    else:
        logger.warning("Config.ini file not found, skipping copy")


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