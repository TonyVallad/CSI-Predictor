"""
Configuration loader for CSI-Predictor.

This module contains configuration loading functionality extracted from the original src/config.py file.
"""

import os
import configparser
from pathlib import Path
from typing import Any, Dict, List
from dotenv import dotenv_values
from src.utils.logging import logger
from .config import Config

class ConfigLoader:
    """
    Configuration loader that handles loading, validation, and merging of configuration
    from multiple sources (.env files and config.ini).
    """
    
    def __init__(self, env_path: str = ".env", ini_path: str = "config.ini"):
        """
        Initialize configuration loader.
        
        Args:
            env_path: Path to .env file
            ini_path: Path to config.ini file
        """
        self.env_path = Path(env_path)
        self.ini_path = Path(ini_path)
        self._env_vars = {}
        self._ini_vars = {}
        self._missing_keys = []
    
    def load_env_vars(self) -> Dict[str, Any]:
        """
        Load environment variables from .env file using dotenv_values.
        
        Returns:
            Dictionary of environment variables
        """
        if self.env_path.exists():
            logger.info(f"Loading environment variables from {self.env_path}")
            env_vars = dotenv_values(self.env_path)
            self._env_vars = dict(env_vars)  # Convert from dotenv dict to regular dict
            logger.debug(f"Loaded {len(self._env_vars)} environment variables")
            return self._env_vars
        else:
            logger.warning(f"Environment file not found: {self.env_path}")
            # Also check system environment variables
            system_env_keys = [
                "DEVICE", "LOAD_DATA_TO_MEMORY", "DATA_SOURCE", "DATA_DIR",
                "MODELS_DIR", "CSV_DIR", "INI_DIR", "GRAPH_DIR", "DEBUG_DIR", 
                "LABELS_CSV", "LABELS_CSV_SEPARATOR"
            ]
            for key in system_env_keys:
                if key in os.environ:
                    self._env_vars[key] = os.environ[key]
            
            if self._env_vars:
                logger.info(f"Using {len(self._env_vars)} system environment variables")
            
            return self._env_vars
    
    def load_ini_vars(self) -> Dict[str, Any]:
        """
        Load configuration from INI file using configparser.
        
        Returns:
            Dictionary of INI file variables
        """
        if self.ini_path.exists():
            logger.info(f"Loading configuration from {self.ini_path}")
            config = configparser.ConfigParser()
            config.read(self.ini_path)
            
            # Flatten INI structure to simple key-value pairs
            ini_vars = {}
            for section_name, section in config.items():
                if section_name != 'DEFAULT':  # Skip DEFAULT section
                    for key, value in section.items():
                        ini_vars[key.upper()] = value
            
            self._ini_vars = ini_vars
            logger.debug(f"Loaded {len(self._ini_vars)} configuration values from INI file")
            return self._ini_vars
        else:
            logger.warning(f"Configuration file not found: {self.ini_path}")
            return {}
    
    def parse_comma_separated_list(self, value: str) -> List[str]:
        """
        Parse comma-separated string into list of strings.
        
        Args:
            value: Comma-separated string
            
        Returns:
            List of trimmed strings (empty list if value is empty)
        """
        if not value or not value.strip():
            return []
        
        # Split by comma and strip whitespace from each item
        items = [item.strip() for item in value.split(',')]
        # Filter out empty strings
        return [item for item in items if item]
    
    def convert_type(self, value: Any, target_type: type) -> Any:
        """
        Convert value to target type with proper handling of string representations.
        
        Args:
            value: Value to convert
            target_type: Target type (int, float, bool, str)
            
        Returns:
            Converted value
        """
        if value is None:
            return None
        
        # Handle string to bool conversion
        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        
        # Handle string to list conversion
        if target_type == list:
            if isinstance(value, str):
                return self.parse_comma_separated_list(value)
            elif isinstance(value, list):
                return value
            else:
                return [str(value)]
        
        # Handle other types
        try:
            return target_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert '{value}' to {target_type.__name__}: {e}")
            return None
    
    def get_config_value(self, key: str, default: Any, target_type: type) -> Any:
        """
        Get configuration value from environment variables or INI file.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            target_type: Target type for conversion
            
        Returns:
            Configuration value
        """
        # Check environment variables first
        if key in self._env_vars:
            value = self._env_vars[key]
            converted = self.convert_type(value, target_type)
            if converted is not None:
                return converted
        
        # Check INI file
        if key in self._ini_vars:
            value = self._ini_vars[key]
            converted = self.convert_type(value, target_type)
            if converted is not None:
                return converted
        
        # Use default value
        if key not in self._missing_keys:
            self._missing_keys.append(key)
            logger.debug(f"Using default value for missing key: {key} = {default}")
        
        return default
    
    def create_config(self) -> Config:
        """
        Create Config instance from loaded environment and INI variables.
        
        Returns:
            Config instance
        """
        # Load environment and INI variables
        self.load_env_vars()
        self.load_ini_vars()
        
        # Create config with loaded values
        config = Config(
            # Environment and Device Settings
            device=self.get_config_value("DEVICE", "cuda", str),
            load_data_to_memory=self.get_config_value("LOAD_DATA_TO_MEMORY", True, bool),
            
            # Data Paths
            data_source=self.get_config_value("DATA_SOURCE", "local", str),
            data_dir=self.get_config_value("DATA_DIR", "/home/pyuser/data/Paradise_Images", str),
            models_dir=self.get_config_value("MODELS_DIR", "./models", str),
            csv_dir=self.get_config_value("CSV_DIR", "/home/pyuser/data/Paradise_CSV", str),
            ini_dir=self.get_config_value("INI_DIR", "./config/", str),
            graph_dir=self.get_config_value("GRAPH_DIR", "./graphs", str),
            debug_dir=self.get_config_value("DEBUG_DIR", "./debug_output", str),
            
            # Labels configuration
            labels_csv=self.get_config_value("LABELS_CSV", "Labeled_Data_RAW.csv", str),
            labels_csv_separator=self.get_config_value("LABELS_CSV_SEPARATOR", ";", str),
            
            # Data Filtering
            excluded_file_ids=self.get_config_value("EXCLUDED_FILE_IDS", [], list),
            
            # Training Hyperparameters
            batch_size=self.get_config_value("BATCH_SIZE", 32, int),
            n_epochs=self.get_config_value("N_EPOCHS", 100, int),
            patience=self.get_config_value("PATIENCE", 10, int),
            learning_rate=self.get_config_value("LEARNING_RATE", 0.001, float),
            optimizer=self.get_config_value("OPTIMIZER", "adam", str),
            dropout_rate=self.get_config_value("DROPOUT_RATE", 0.5, float),
            weight_decay=self.get_config_value("WEIGHT_DECAY", 0.01, float),
            
            # Model Configuration
            model_arch=self.get_config_value("MODEL_ARCH", "resnet50", str),
            use_official_processor=self.get_config_value("USE_OFFICIAL_PROCESSOR", False, bool),
            zone_focus_method=self.get_config_value("ZONE_FOCUS_METHOD", "masking", str),
            
            # Zone Masking Configuration
            use_segmentation_masking=self.get_config_value("USE_SEGMENTATION_MASKING", True, bool),
            masking_strategy=self.get_config_value("MASKING_STRATEGY", "attention", str),
            attention_strength=self.get_config_value("ATTENTION_STRENGTH", 0.7, float),
            masks_path=self.get_config_value("MASKS_PATH", "/home/pyuser/data/Paradise_Masks", str),
            
            # Image Format Configuration
            image_format=self.get_config_value("IMAGE_FORMAT", "nifti", str),
            image_extension=self.get_config_value("IMAGE_EXTENSION", ".nii.gz", str),
            
            # Normalization Strategy Configuration
            normalization_strategy=self.get_config_value("NORMALIZATION_STRATEGY", "medical", str),
            custom_mean=self.get_config_value("CUSTOM_MEAN", None, list),
            custom_std=self.get_config_value("CUSTOM_STD", None, list)
        )
        
        # Log missing keys
        if self._missing_keys:
            logger.info(f"Missing configuration keys (using defaults): {', '.join(self._missing_keys)}")
        
        return config

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 