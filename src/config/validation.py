"""
Configuration validation for CSI-Predictor.

This module contains configuration validation functionality extracted from the original src/config.py file.
"""

import os
from pathlib import Path
from typing import Any
from ...utils.logging import logger
from .config import Config

def validate_config(config: Config) -> None:
    """
    Validate configuration values and raise errors for invalid settings.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    
    # Validate device
    if config.device not in ["cuda", "cpu"]:
        errors.append(f"Invalid device: {config.device}. Must be 'cuda' or 'cpu'")
    
    # Validate data paths
    if not os.path.exists(config.data_dir):
        errors.append(f"Data directory does not exist: {config.data_dir}")
    
    if not os.path.exists(config.csv_dir):
        errors.append(f"CSV directory does not exist: {config.csv_dir}")
    
    # Validate CSV file
    csv_path = os.path.join(config.csv_dir, config.labels_csv)
    if not os.path.exists(csv_path):
        errors.append(f"CSV file does not exist: {csv_path}")
    
    # Validate training hyperparameters
    if config.batch_size <= 0:
        errors.append(f"Invalid batch_size: {config.batch_size}. Must be positive")
    
    if config.n_epochs <= 0:
        errors.append(f"Invalid n_epochs: {config.n_epochs}. Must be positive")
    
    if config.patience <= 0:
        errors.append(f"Invalid patience: {config.patience}. Must be positive")
    
    if config.learning_rate <= 0:
        errors.append(f"Invalid learning_rate: {config.learning_rate}. Must be positive")
    
    if config.dropout_rate < 0 or config.dropout_rate > 1:
        errors.append(f"Invalid dropout_rate: {config.dropout_rate}. Must be between 0 and 1")
    
    if config.weight_decay < 0:
        errors.append(f"Invalid weight_decay: {config.weight_decay}. Must be non-negative")
    
    # Validate optimizer
    valid_optimizers = ["adam", "adamw", "sgd"]
    if config.optimizer.lower() not in valid_optimizers:
        errors.append(f"Invalid optimizer: {config.optimizer}. Must be one of {valid_optimizers}")
    
    # Validate model architecture
    valid_architectures = ["resnet50", "densenet121", "custom_cnn", "raddino"]
    if config.model_arch.lower() not in valid_architectures:
        errors.append(f"Invalid model_arch: {config.model_arch}. Must be one of {valid_architectures}")
    
    # Validate zone focus method
    valid_zone_focus_methods = ["masking", "spatial_reduction"]
    if config.zone_focus_method.lower() not in valid_zone_focus_methods:
        errors.append(f"Invalid zone_focus_method: {config.zone_focus_method}. Must be one of {valid_zone_focus_methods}")
    
    # Validate masking strategy
    valid_masking_strategies = ["zero", "attention"]
    if config.masking_strategy.lower() not in valid_masking_strategies:
        errors.append(f"Invalid masking_strategy: {config.masking_strategy}. Must be one of {valid_masking_strategies}")
    
    # Validate attention strength
    if config.attention_strength < 0 or config.attention_strength > 1:
        errors.append(f"Invalid attention_strength: {config.attention_strength}. Must be between 0 and 1")
    
    # Validate image format
    if config.image_format.lower() != "nifti":
        errors.append(f"Invalid image_format: {config.image_format}. Only 'nifti' is supported")
    
    # Validate normalization strategy
    valid_normalization_strategies = ["imagenet", "medical", "simple", "custom"]
    if config.normalization_strategy.lower() not in valid_normalization_strategies:
        errors.append(f"Invalid normalization_strategy: {config.normalization_strategy}. Must be one of {valid_normalization_strategies}")
    
    # Validate custom normalization parameters
    if config.normalization_strategy.lower() == "custom":
        if config.custom_mean is None or config.custom_std is None:
            errors.append("custom_mean and custom_std must be provided when normalization_strategy is 'custom'")
        elif len(config.custom_mean) != 3 or len(config.custom_std) != 3:
            errors.append("custom_mean and custom_std must be lists of length 3")
    
    # Validate masks path if using segmentation masking
    if config.use_segmentation_masking:
        if not os.path.exists(config.masks_path):
            errors.append(f"Masks directory does not exist: {config.masks_path}")
    
    # Check for errors
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info("Configuration validation passed")

def validate_paths(config: Config) -> None:
    """
    Validate that all required paths exist and are accessible.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If any required path is invalid
    """
    paths_to_check = [
        ("data_dir", config.data_dir),
        ("csv_dir", config.csv_dir),
        ("models_dir", config.models_dir),
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            logger.warning(f"{name} does not exist: {path}")
            # Try to create the directory
            try:
                os.makedirs(path, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                logger.error(f"Could not create directory {path}: {e}")

def validate_file_permissions(config: Config) -> None:
    """
    Validate that the application has proper permissions to read/write files.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        PermissionError: If permissions are insufficient
    """
    # Check read permissions for data directories
    if not os.access(config.data_dir, os.R_OK):
        raise PermissionError(f"No read permission for data directory: {config.data_dir}")
    
    if not os.access(config.csv_dir, os.R_OK):
        raise PermissionError(f"No read permission for CSV directory: {config.csv_dir}")
    
    # Check write permissions for output directories
    if not os.access(config.models_dir, os.W_OK):
        raise PermissionError(f"No write permission for models directory: {config.models_dir}")
    
    if not os.access(config.graph_dir, os.W_OK):
        raise PermissionError(f"No write permission for graph directory: {config.graph_dir}")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 