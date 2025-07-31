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
from src.utils.logging import logger
from .config import Config
from .config_loader import ConfigLoader
from .validation import validate_config, validate_paths, validate_file_permissions

# Singleton instance
_config_instance: Optional[Config] = None

# Backward compatibility: create cfg property that points to the singleton instance
class ConfigProxy:
    """Proxy class for backward compatibility with cfg variable."""
    
    def __getattr__(self, name):
        """Get attribute from the singleton config instance."""
        return getattr(get_config(), name)
    
    def __setattr__(self, name, value):
        """Set attribute on the singleton config instance."""
        setattr(get_config(), name, value)
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return getattr(get_config(), key)
    
    def __setitem__(self, key, value):
        """Allow dictionary-style assignment."""
        setattr(get_config(), key, value)

# Create cfg variable for backward compatibility
cfg = ConfigProxy()

def get_config(env_path: str = ".env", ini_path: str = None, force_reload: bool = False) -> Config:
    """
    Get singleton configuration instance.
    
    Args:
        env_path: Path to .env file
        ini_path: Path to config.ini file (if None, will be determined from .env INI_DIR)
        force_reload: Whether to force reload configuration
        
    Returns:
        Singleton Config instance
    """
    global _config_instance
    
    if _config_instance is None or force_reload:
        logger.info("Initializing configuration...")
        
        # First, load environment variables to get INI_DIR
        temp_loader = ConfigLoader(env_path, "dummy.ini")  # Temporary loader just for env vars
        env_vars = temp_loader.load_env_vars()
        
        # Determine ini_path from INI_DIR if not provided
        if ini_path is None:
            ini_dir = env_vars.get("INI_DIR", "")
            if ini_dir:
                # If INI_DIR is set, use it
                ini_path = os.path.join(ini_dir, "config.ini")
                logger.info(f"Using INI_DIR from .env: {ini_path}")
            else:
                # If INI_DIR is empty, use DATA_DIR/config (following the same pattern as other paths)
                data_dir = env_vars.get("DATA_DIR", "./data")
                ini_path = os.path.join(data_dir, "config", "config.ini")
                logger.info(f"INI_DIR not set in .env, using DATA_DIR/config: {ini_path}")
                
                # If the DATA_DIR/config path doesn't exist, try fallback paths
                if not os.path.exists(ini_path):
                    logger.warning(f"Config file not found at {ini_path}, trying fallback paths...")
                    possible_paths = [
                        "config/config.ini",
                        "../config/config.ini",
                        "../../config/config.ini",
                        "src/../config/config.ini",
                    ]
                    
                    for path in possible_paths:
                        if os.path.exists(path):
                            ini_path = path
                            logger.info(f"Using fallback path: {ini_path}")
                            break
                    else:
                        # If not found, use the default
                        ini_path = "config/config.ini"
                        logger.warning(f"No config file found, using default: {ini_path}")
        
        # Create the actual config loader with the determined paths
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
    Copy the current configuration to a timestamped file in the config directory.
    This is called at the start of training to preserve the exact configuration used.
    """
    config = get_config()
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"config_{timestamp}.ini"
    
    # Create the config directory if it doesn't exist
    os.makedirs(config.ini_dir, exist_ok=True)
    
    # Create the timestamped config file
    timestamped_path = Path(config.ini_dir) / timestamped_filename
    
    # Create a new config parser
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
    
    # Add data section
    new_config.add_section("DATA")
    excluded_ids_str = ",".join(config.excluded_file_ids) if config.excluded_file_ids else ""
    new_config.set("DATA", "EXCLUDED_FILE_IDS", excluded_ids_str)
    
    # Add zones section
    new_config.add_section("ZONES")
    new_config.set("ZONES", "USE_SEGMENTATION_MASKING", str(config.use_segmentation_masking))
    new_config.set("ZONES", "MASKING_STRATEGY", config.masking_strategy)
    new_config.set("ZONES", "ATTENTION_STRENGTH", str(config.attention_strength))
    
    # Write the config file
    with open(timestamped_path, 'w') as configfile:
        new_config.write(configfile)
    
    logger.info(f"Configuration saved to: {timestamped_path}")

def create_default_env_file(env_path: str = ".env") -> None:
    """
    Create a default .env file with all the new path variables.
    
    Args:
        env_path: Path where to create the .env file
    """
    env_content = """# CSI-Predictor Configuration

# Discord Webhook for model results
DISCORD_WEBHOOK_URL=

# Device configuration
DEVICE=cuda

# Data loading configuration
LOAD_DATA_TO_MEMORY=True

# Data source and paths
DATA_DIR=./data
INI_DIR=  # Leave empty to use DATA_DIR/config, or specify custom path
CSV_DIR=  # Leave empty to use DATA_DIR/csv, or specify custom path
DICOM_DIR=  # Leave empty to use DATA_DIR/dicom, or specify custom path
DICOM_HIST_DIR=  # Leave empty to use DATA_DIR/dicom_hist, or specify custom path
NIFTI_HIST_DIR=  # Leave empty to use DATA_DIR/nifti_hist, or specify custom path
NIFTI_DIR=  # Leave empty to use DATA_DIR/nifti, or specify custom path
PNG_DIR=  # Leave empty to use DATA_DIR/png, or specify custom path
MODELS_DIR=  # Leave empty to use DATA_DIR/models, or specify custom path
GRAPH_DIR=  # Leave empty to use DATA_DIR/graphs, or specify custom path
DEBUG_DIR=  # Leave empty to use DATA_DIR/debug, or specify custom path
MASKS_DIR=  # Leave empty to use DATA_DIR/masks, or specify custom path
LOGS_DIR=  # Leave empty to use DATA_DIR/logs, or specify custom path
RUNS_DIR=  # Leave empty to use DATA_DIR/runs, or specify custom path
EVALUATION_DIR=  # Leave empty to use DATA_DIR/evaluations, or specify custom path
WANDB_DIR=  # Leave empty to use DATA_DIR/wandb, or specify custom path

# Labels configuration
LABELS_CSV=Labeled_Data_RAW.csv
LABELS_CSV_SEPARATOR=;

# Training Parameters (can be overridden by config.ini)
BATCH_SIZE=32
N_EPOCHS=100
LEARNING_RATE=0.001
MODEL_ARCH=resnet50
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info(f"Default .env file created at: {env_path}")

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 