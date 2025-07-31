"""
Legacy configuration module for CSI-Predictor.

This module provides backward compatibility for the old configuration system.
New code should use src.config instead.
"""

import os
import configparser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import dotenv_values
from loguru import logger

# Import the new config system
from src.config import get_config as get_new_config, Config as NewConfig

@dataclass(frozen=True)
class Config:
    """
    Legacy configuration dataclass for backward compatibility.
    
    This dataclass is frozen (immutable) to prevent accidental modification
    of configuration values during runtime.
    
    Attributes:
        # Environment and Device Settings
        device: Device for computation (cuda/cpu)
        load_data_to_memory: Whether to load all data into memory
        
        # Data Paths
        data_source: Source of data (local/remote)
        data_dir: Directory containing images (legacy - use nifti_dir)
        models_dir: Directory for saving/loading models
        csv_dir: Directory containing CSV files
        ini_dir: Directory containing config files
        graph_dir: Directory for saving graphs and visualizations
        
        # Labels configuration
        labels_csv: CSV filename (not full path)
        labels_csv_separator: CSV separator character
        
        # Training Hyperparameters
        batch_size: Training batch size
        n_epochs: Number of training epochs
        patience: Early stopping patience
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type (adam/adamw/sgd)
        
        # Model Configuration
        model_arch: Model architecture name
        use_official_processor: Whether to use official RadDINO processor
        
        # Image Format Configuration (V2.0 - NIFTI Support)
        image_format: str = "nifti"  # Image format (nifti only)
        image_extension: str = ".nii.gz"  # File extension for images
        
        # Normalization Strategy Configuration
        normalization_strategy: str = "medical"  # Options: "imagenet", "medical", "simple", "custom"
        custom_mean: List[float] = None  # Custom mean values for normalization (if strategy="custom")
        custom_std: List[float] = None   # Custom std values for normalization (if strategy="custom")
        
        # Internal
        _env_vars: Dict[str, Any] = field(default_factory=dict, init=False)
        _ini_vars: Dict[str, Any] = field(default_factory=dict, init=False)
        _missing_keys: List[str] = field(default_factory=list, init=False)
    """
    
    # Environment and Device Settings
    device: str = "cuda"
    load_data_to_memory: bool = True
    
    # Data Paths
    data_source: str = "local"
    data_dir: str = "/home/pyuser/data/Paradise_Images"  # Legacy - use nifti_dir
    models_dir: str = "./models"
    csv_dir: str = "/home/pyuser/data/Paradise_CSV"
    ini_dir: str = "./config/"
    graph_dir: str = "./graphs"  # Directory for saving graphs and visualizations
    debug_dir: str = "./debug_output"  # Directory for debug visualizations
    
    # Labels configuration
    labels_csv: str = "Labeled_Data_RAW.csv"
    labels_csv_separator: str = ";"
    
    # Data Filtering
    excluded_file_ids: List[str] = field(default_factory=list)
    
    # Training Hyperparameters
    batch_size: int = 32
    n_epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    dropout_rate: float = 0.5
    weight_decay: float = 0.01
    
    # Model Configuration
    model_arch: str = "resnet50"
    use_official_processor: bool = False
    zone_focus_method: str = "masking"  # "masking" or "spatial_reduction"
    
    # Zone Masking Configuration
    use_segmentation_masking: bool = True
    masking_strategy: str = "attention"  # "zero" or "attention"
    attention_strength: float = 0.7
    masks_path: str = "/home/pyuser/data/Paradise_Masks"  # Legacy - use masks_dir
    
    # Image Format Configuration (V2.0 - NIFTI Support)
    image_format: str = "nifti"  # Image format (nifti only)
    image_extension: str = ".nii.gz"  # File extension for images
    
    # Normalization Strategy Configuration
    normalization_strategy: str = "medical"  # Options: "imagenet", "medical", "simple", "custom"
    custom_mean: List[float] = None  # Custom mean values for normalization (if strategy="custom")
    custom_std: List[float] = None   # Custom std values for normalization (if strategy="custom")
    
    # Internal
    _env_vars: Dict[str, Any] = field(default_factory=dict, init=False)
    _ini_vars: Dict[str, Any] = field(default_factory=dict, init=False)
    _missing_keys: List[str] = field(default_factory=list, init=False)
    
    @property
    def data_path(self) -> str:
        """Legacy property for backward compatibility."""
        return self.data_dir
    
    @property
    def csv_path(self) -> str:
        """Construct full path to labels CSV file."""
        return f"{self.csv_dir}/{self.labels_csv}"
    
    @property
    def models_folder(self) -> str:
        """Legacy property for backward compatibility."""
        return self.models_dir
    
    def get_model_path(self, model_name: str, extension: str = "pth") -> str:
        """
        Get the full path for a model file.
        
        Args:
            model_name: Name of the model (without extension)
            extension: File extension (default: pth)
            
        Returns:
            Full path to the model file
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        return f"{self.models_dir}/{model_name}{extension}"

class ConfigLoader:
    """
    Legacy configuration loader for backward compatibility.
    
    This class provides backward compatibility for the old configuration system.
    New code should use src.config.ConfigLoader instead.
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
                "NIFTI_DIR", "MODELS_DIR", "CSV_DIR", "INI_DIR", "PNG_DIR", 
                "GRAPH_DIR", "DEBUG_DIR", "MASKS_DIR", "LOGS_DIR", "RUNS_DIR",
                "EVALUATION_DIR", "WANDB_DIR", "LABELS_CSV", "LABELS_CSV_SEPARATOR"
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
                        # Strip comments from values (everything after #)
                        clean_value = value.split('#')[0].strip()
                        ini_vars[key.upper()] = clean_value
            
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
            List of strings
        """
        if not value or value.strip() == "":
            return []
        
        # Split by comma and strip whitespace
        items = [item.strip() for item in value.split(',')]
        # Remove empty items
        return [item for item in items if item]
    
    def convert_type(self, value: Any, target_type: type) -> Any:
        """
        Convert value to target type with error handling.
        
        Args:
            value: Value to convert
            target_type: Target type
            
        Returns:
            Converted value or None if conversion fails
        """
        if value is None:
            return None
        
        try:
            if target_type == bool:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif target_type == list:
                if isinstance(value, str):
                    return self.parse_comma_separated_list(value)
                elif isinstance(value, list):
                    return value
                else:
                    return [str(value)]
            else:
                return target_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert {value} to {target_type.__name__}: {e}")
            return None
    
    def get_config_value(self, key: str, default: Any, target_type: type) -> Any:
        """
        Get configuration value with type conversion and fallback logic.
        
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
        Create and validate configuration by merging all sources.
        
        Returns:
            Immutable Config instance
        """
        # Load from all sources
        self.load_env_vars()
        self.load_ini_vars()
        
        # Get the master data directory first
        data_dir = self.get_config_value("DATA_DIR", "/home/pyuser/data/Paradise", str)
        
        # Resolve all path variables
        nifti_dir = self.resolve_path("NIFTI_DIR", "nifti", data_dir)
        csv_dir = self.resolve_path("CSV_DIR", "csv", data_dir)
        ini_dir = self.resolve_path("INI_DIR", "config", data_dir)
        png_dir = self.resolve_path("PNG_DIR", "png", data_dir)
        graph_dir = self.resolve_path("GRAPH_DIR", "graphs", data_dir)
        debug_dir = self.resolve_path("DEBUG_DIR", "debug", data_dir)
        masks_dir = self.resolve_path("MASKS_DIR", "masks", data_dir)
        runs_dir = self.resolve_path("RUNS_DIR", "runs", data_dir)
        evaluation_dir = self.resolve_path("EVALUATION_DIR", "evaluations", data_dir)
        
        # Some paths default to relative paths if not set
        models_dir = self.get_config_value("MODELS_DIR", "./models", str)
        logs_dir = self.get_config_value("LOGS_DIR", "./logs", str)
        wandb_dir = self.get_config_value("WANDB_DIR", "./wandb", str)
        
        # Create config with type-safe value extraction
        config = Config(
            # Environment and Device Settings
            device=self.get_config_value("DEVICE", "cuda", str),
            load_data_to_memory=self.get_config_value("LOAD_DATA_TO_MEMORY", True, bool),
            
            # Data Paths
            data_source=self.get_config_value("DATA_SOURCE", "local", str),
            data_dir=nifti_dir,  # Legacy compatibility - map nifti_dir to data_dir
            models_dir=models_dir,
            csv_dir=csv_dir,
            ini_dir=ini_dir,
            graph_dir=graph_dir,
            debug_dir=debug_dir,
            labels_csv=self.get_config_value("LABELS_CSV", "Labeled_Data_RAW.csv", str),
            labels_csv_separator=self.get_config_value("LABELS_CSV_SEPARATOR", ";", str),
            
            # Data Filtering
            excluded_file_ids=self.parse_comma_separated_list(
                self.get_config_value("EXCLUDED_FILE_IDS", "", str)
            ),
            
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
            masks_path=masks_dir,  # Legacy compatibility - map masks_dir to masks_path
            
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
    
    def resolve_path(self, path_key: str, default_subfolder: str, data_dir: str) -> str:
        """
        Resolve a path variable. If the path is empty or not set, use DATA_DIR + subfolder.
        If the path is set, use it as-is.
        
        Args:
            path_key: The environment variable key for the path
            default_subfolder: The subfolder name to append to DATA_DIR if path is empty
            data_dir: The master data directory path
            
        Returns:
            Resolved path
        """
        # Get the path value from environment variables
        path_value = self._env_vars.get(path_key, "")
        
        # If path is empty or not set, use DATA_DIR + subfolder
        if not path_value or path_value.strip() == "":
            resolved_path = os.path.join(data_dir, default_subfolder)
            logger.debug(f"Path {path_key} not set, using: {resolved_path}")
            return resolved_path
        else:
            logger.debug(f"Path {path_key} set to: {path_value}")
            return path_value
    
    def copy_config_with_timestamp(self, config: Config) -> None:
        """
        Copy resolved configuration with timestamp when training starts.
        This creates a snapshot of the actual configuration used for training.
        """
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

# Legacy function for backward compatibility
def get_config(env_path: str = ".env", ini_path: str = "config.ini") -> Config:
    """
    Legacy function to get configuration for backward compatibility.
    
    Args:
        env_path: Path to .env file
        ini_path: Path to config.ini file
        
    Returns:
        Config instance
    """
    logger.warning("Using legacy config system. Consider migrating to src.config.get_config()")
    loader = ConfigLoader(env_path, ini_path)
    return loader.create_config()

# Singleton instance - import this in other modules
cfg = get_config()

# Export main components
__all__ = ['Config', 'ConfigLoader', 'get_config', 'cfg']