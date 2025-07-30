"""
Configuration package for CSI-Predictor.

This module contains the main configuration functionality extracted from the original src/config.py file.
"""

import os
import configparser
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from ...utils.logging import logger
from .config import Config
from .config_loader import ConfigLoader
from .validation import validate_config, validate_paths, validate_file_permissions

# Singleton instance
_config_instance: Optional[Config] = None

def get_config(env_path: str = ".env", ini_path: str = "config.ini", force_reload: bool = False) -> Config:
    """
    Get singleton configuration instance.
    
    Args:
        env_path: Path to .env file
        ini_path: Path to config.ini file
        force_reload: Whether to force reload configuration
        
    Returns:
        Singleton Config instance
    """
    global _config_instance
    
    if _config_instance is None or force_reload:
        logger.info("Initializing configuration...")
        loader = ConfigLoader(env_path, ini_path)
        _config_instance = loader.create_config()
        
        # Validate configuration
        try:
            validate_config(_config_instance)
            validate_paths(_config_instance)
            validate_file_permissions(_config_instance)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        
        logger.info("Configuration initialized successfully")
    
    return _config_instance

def copy_config_on_training_start() -> None:
    """
    Copy resolved configuration with timestamp when training starts.
    This creates a snapshot of the actual configuration used for training.
    """
    config = get_config()
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"config_resolved_{timestamp}.ini"
    timestamped_path = Path(config.ini_dir) / timestamped_filename
    
    # Ensure config directory exists
    os.makedirs(config.ini_dir, exist_ok=True)
    
    # Create new config parser
    new_config = configparser.ConfigParser()
    
    # Add training section
    new_config.add_section("TRAINING")
    new_config.set("TRAINING", "BATCH_SIZE", str(config.batch_size))
    new_config.set("TRAINING", "N_EPOCHS", str(config.n_epochs))
    new_config.set("TRAINING", "PATIENCE", str(config.patience))
    new_config.set("TRAINING", "LEARNING_RATE", str(config.learning_rate))
    new_config.set("TRAINING", "OPTIMIZER", config.optimizer)
    new_config.set("TRAINING", "DROPOUT_RATE", str(config.dropout_rate))
    new_config.set("TRAINING", "WEIGHT_DECAY", str(config.weight_decay))
    
    # Add model section
    new_config.add_section("MODEL")
    new_config.set("MODEL", "MODEL_ARCH", config.model_arch)
    new_config.set("MODEL", "USE_OFFICIAL_PROCESSOR", str(config.use_official_processor))
    new_config.set("MODEL", "ZONE_FOCUS_METHOD", config.zone_focus_method)
    
    # Add data section
    new_config.add_section("DATA")
    excluded_ids_str = ",".join(config.excluded_file_ids) if config.excluded_file_ids else ""
    new_config.set("DATA", "EXCLUDED_FILE_IDS", excluded_ids_str)
    
    # Add zones section
    new_config.add_section("ZONES")
    new_config.set("ZONES", "USE_SEGMENTATION_MASKING", str(config.use_segmentation_masking))
    new_config.set("ZONES", "MASKING_STRATEGY", config.masking_strategy)
    new_config.set("ZONES", "ATTENTION_STRENGTH", str(config.attention_strength))
    new_config.set("ZONES", "MASKS_PATH", config.masks_path)
    
    # Add image format section
    new_config.add_section("IMAGE_FORMAT")
    new_config.set("IMAGE_FORMAT", "IMAGE_FORMAT", config.image_format)
    new_config.set("IMAGE_FORMAT", "IMAGE_EXTENSION", config.image_extension)
    
    # Add normalization section
    new_config.add_section("NORMALIZATION")
    new_config.set("NORMALIZATION", "NORMALIZATION_STRATEGY", config.normalization_strategy)
    if config.normalization_strategy.lower() == "custom":
        new_config.set("NORMALIZATION", "CUSTOM_MEAN", ",".join(map(str, config.custom_mean)))
        new_config.set("NORMALIZATION", "CUSTOM_STD", ",".join(map(str, config.custom_std)))
    
    # Add environment section with resolved paths
    new_config.add_section("ENVIRONMENT")
    new_config.set("ENVIRONMENT", "DEVICE", config.device)
    new_config.set("ENVIRONMENT", "DATA_SOURCE", config.data_source)
    new_config.set("ENVIRONMENT", "DATA_DIR", config.data_dir)
    new_config.set("ENVIRONMENT", "MODELS_DIR", config.models_dir)
    new_config.set("ENVIRONMENT", "CSV_DIR", config.csv_dir)
    new_config.set("ENVIRONMENT", "INI_DIR", config.ini_dir)
    new_config.set("ENVIRONMENT", "GRAPH_DIR", config.graph_dir)
    new_config.set("ENVIRONMENT", "DEBUG_DIR", config.debug_dir)
    new_config.set("ENVIRONMENT", "LABELS_CSV", config.labels_csv)
    new_config.set("ENVIRONMENT", "LABELS_CSV_SEPARATOR", config.labels_csv_separator)
    new_config.set("ENVIRONMENT", "LOAD_DATA_TO_MEMORY", str(config.load_data_to_memory))
    
    # Write timestamped config
    try:
        with open(timestamped_path, 'w') as f:
            new_config.write(f)
        logger.info(f"Copied resolved configuration to {timestamped_path}")
    except Exception as e:
        logger.error(f"Failed to copy configuration: {e}")

# Singleton instance - import this in other modules
cfg = get_config()

# Export main components
__all__ = ['Config', 'ConfigLoader', 'get_config', 'copy_config_on_training_start', 'cfg', 'validate_config', 'validate_paths', 'validate_file_permissions']

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 