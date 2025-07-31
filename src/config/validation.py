"""
Configuration validation for CSI-Predictor.

This module contains validation functions for configuration settings.
"""

import os
from typing import List, Tuple
from src.utils.logging import logger
from .config import Config

def validate_config(config: Config) -> None:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration instance to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    errors = []
    
    # Validate device
    if config.device not in ["cuda", "cpu", "mps"]:
        errors.append(f"Invalid device: {config.device}. Must be 'cuda', 'cpu', or 'mps'")
    
    # Validate batch size
    if config.batch_size <= 0:
        errors.append(f"Batch size must be positive, got: {config.batch_size}")
    
    # Validate learning rate
    if config.learning_rate <= 0:
        errors.append(f"Learning rate must be positive, got: {config.learning_rate}")
    
    # Validate optimizer
    valid_optimizers = ["adam", "adamw", "sgd"]
    if config.optimizer.lower() not in valid_optimizers:
        errors.append(f"Invalid optimizer: {config.optimizer}. Must be one of: {valid_optimizers}")
    
    # Validate model architecture
    valid_architectures = ["resnet50", "chexnet", "custom1", "raddino"]
    if config.model_arch.lower() not in valid_architectures:
        errors.append(f"Invalid model architecture: {config.model_arch}. Must be one of: {valid_architectures}")
    
    # Validate zone focus method
    valid_zone_methods = ["masking", "spatial_reduction"]
    if config.zone_focus_method.lower() not in valid_zone_methods:
        errors.append(f"Invalid zone focus method: {config.zone_focus_method}. Must be one of: {valid_zone_methods}")
    
    # Validate masking strategy
    valid_masking_strategies = ["zero", "attention"]
    if config.masking_strategy.lower() not in valid_masking_strategies:
        errors.append(f"Invalid masking strategy: {config.masking_strategy}. Must be one of: {valid_masking_strategies}")
    
    # Validate attention strength
    if not 0.0 <= config.attention_strength <= 1.0:
        errors.append(f"Attention strength must be between 0.0 and 1.0, got: {config.attention_strength}")
    
    # Validate image format
    if config.image_format.lower() != "nifti":
        errors.append(f"Only NIFTI format is supported, got: {config.image_format}")
    
    # Validate normalization strategy
    valid_normalization_strategies = ["imagenet", "medical", "simple", "custom"]
    if config.normalization_strategy.lower() not in valid_normalization_strategies:
        errors.append(f"Invalid normalization strategy: {config.normalization_strategy}. Must be one of: {valid_normalization_strategies}")
    
    # Validate custom normalization if strategy is custom
    if config.normalization_strategy.lower() == "custom":
        if config.custom_mean is None or config.custom_std is None:
            errors.append("Custom normalization strategy requires CUSTOM_MEAN and CUSTOM_STD values")
        elif len(config.custom_mean) != 3 or len(config.custom_std) != 3:
            errors.append("CUSTOM_MEAN and CUSTOM_STD must have exactly 3 values (RGB)")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))

def validate_paths(config: Config) -> None:
    """
    Validate that required paths exist and are accessible.
    
    Args:
        config: Configuration instance to validate
        
    Raises:
        FileNotFoundError: If required paths don't exist
    """
    errors = []
    
    # Required paths that must exist
    required_paths = [
        ("nifti_dir", config.nifti_dir),
        ("csv_dir", config.csv_dir),
    ]
    
    # Check if masks directory exists when segmentation masking is enabled
    if config.use_segmentation_masking:
        required_paths.append(("masks_dir", config.masks_dir))
    
    for path_name, path_value in required_paths:
        if not os.path.exists(path_value):
            errors.append(f"{path_name} does not exist: {path_value}")
    
    if errors:
        raise FileNotFoundError(f"Path validation failed:\n" + "\n".join(f"- {error}" for error in errors))

def validate_file_permissions(config: Config) -> None:
    """
    Validate file permissions for required paths.
    
    Args:
        config: Configuration instance to validate
        
    Raises:
        PermissionError: If required permissions are missing
    """
    # Check read permissions for data directories
    read_paths = [
        ("nifti_dir", config.nifti_dir),
        ("csv_dir", config.csv_dir),
    ]
    
    if config.use_segmentation_masking:
        read_paths.append(("masks_dir", config.masks_dir))
    
    for path_name, path_value in read_paths:
        if not os.access(path_value, os.R_OK):
            raise PermissionError(f"No read permission for {path_name}: {path_value}")
    
    # Check write permissions for output directories
    write_paths = [
        ("models_dir", config.models_dir),
        ("graph_dir", config.graph_dir),
        ("logs_dir", config.logs_dir),
    ]
    
    for path_name, path_value in write_paths:
        if not os.access(path_value, os.W_OK):
            raise PermissionError(f"No write permission for {path_name}: {path_value}")

def validate_csv_file(config: Config) -> None:
    """
    Validate that the CSV file exists and is readable.
    
    Args:
        config: Configuration instance to validate
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        PermissionError: If CSV file is not readable
    """
    csv_path = os.path.join(config.csv_dir, config.labels_csv)
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    
    if not os.access(csv_path, os.R_OK):
        raise PermissionError(f"No read permission for CSV file: {csv_path}")

def get_validation_summary(config: Config) -> List[Tuple[str, str]]:
    """
    Get a summary of configuration validation results.
    
    Args:
        config: Configuration instance to validate
        
    Returns:
        List of (status, message) tuples
    """
    summary = []
    
    # Check required paths
    required_paths = [
        ("nifti_dir", config.nifti_dir),
        ("csv_dir", config.csv_dir),
    ]
    
    if config.use_segmentation_masking:
        required_paths.append(("masks_dir", config.masks_dir))
    
    for path_name, path_value in required_paths:
        if os.path.exists(path_value):
            summary.append(("✅", f"{path_name}: {path_value}"))
        else:
            summary.append(("❌", f"{path_name}: {path_value} (does not exist)"))
    
    # Check CSV file
    csv_path = os.path.join(config.csv_dir, config.labels_csv)
    if os.path.exists(csv_path):
        summary.append(("✅", f"CSV file: {csv_path}"))
    else:
        summary.append(("❌", f"CSV file: {csv_path} (does not exist)"))
    
    # Check write permissions for output directories
    write_paths = [
        ("models_dir", config.models_dir),
        ("graph_dir", config.graph_dir),
        ("logs_dir", config.logs_dir),
    ]
    
    for path_name, path_value in write_paths:
        if os.access(path_value, os.W_OK):
            summary.append(("✅", f"{path_name}: {path_value} (writable)"))
        else:
            summary.append(("⚠️", f"{path_name}: {path_value} (not writable)"))
    
    return summary

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 