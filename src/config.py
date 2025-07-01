"""
Centralized Configuration Loader for CSI-Predictor.

This module provides a singleton configuration system that:
- Loads environment variables from .env files using dotenv_values
- Loads configuration from INI files using configparser
- Merges and validates configuration into an immutable dataclass
- Provides automatic type conversion for int, float, bool values
- Logs missing keys and validation errors using loguru
- Copies resolved config.ini with timestamp when training starts

Usage:
    from src.config import cfg
    
    # Access configuration values
    print(cfg.batch_size)  # Automatically converted to int
    print(cfg.learning_rate)  # Automatically converted to float
    print(cfg.device)  # String value
"""

import os
import configparser
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dotenv import dotenv_values
from loguru import logger


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration dataclass that holds all application settings.
    
    This dataclass is frozen (immutable) to prevent accidental modification
    of configuration values during runtime.
    
    Attributes:
        # Environment and Device Settings
        device: Device for computation (cuda/cpu)
        load_data_to_memory: Whether to load all data into memory
        
        # Data Paths
        data_source: Source of data (local/remote)
        data_dir: Directory containing images
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
        
        # Internal
        _env_vars: Environment variables dict
        _ini_vars: INI file variables dict
        _missing_keys: List of missing configuration keys
    """
    
    # Environment and Device Settings
    device: str = "cuda"
    load_data_to_memory: bool = True
    
    # Data Paths
    data_source: str = "local"
    data_dir: str = "/home/pyuser/data/Paradise_Images"
    models_dir: str = "./models"
    csv_dir: str = "/home/pyuser/data/Paradise_CSV"
    ini_dir: str = "./config/"
    graph_dir: str = "./graphs"  # Directory for saving graphs and visualizations
    
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
    
    # Model Configuration
    model_arch: str = "resnet50"
    use_official_processor: bool = False  # Whether to use official RadDINO processor
    zone_focus_method: str = "masking"  # "masking" or "spatial_reduction"
    
    # Zone Masking Configuration
    use_segmentation_masking: bool = True
    masking_strategy: str = "attention"  # "zero" or "attention"
    attention_strength: float = 0.7
    masks_path: str = "/home/pyuser/data/Paradise_Masks"
    
    # Internal fields (not for external configuration)
    _env_vars: Dict[str, Any] = field(default_factory=dict, repr=False)
    _ini_vars: Dict[str, Any] = field(default_factory=dict, repr=False)
    _missing_keys: List[str] = field(default_factory=list, repr=False)
    
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
            
        # Unit test stub:
        # def test_load_env_vars():
        #     loader = ConfigLoader()
        #     # Create temporary .env file
        #     with open(".env.test", "w") as f:
        #         f.write("DEVICE=cpu\nBATCH_SIZE=64\n")
        #     loader.env_path = Path(".env.test")
        #     env_vars = loader.load_env_vars()
        #     assert env_vars["DEVICE"] == "cpu"
        #     assert env_vars["BATCH_SIZE"] == "64"
        #     os.remove(".env.test")
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
                "MODELS_DIR", "CSV_DIR", "INI_DIR", "LABELS_CSV", "LABELS_CSV_SEPARATOR"
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
            
        # Unit test stub:
        # def test_load_ini_vars():
        #     loader = ConfigLoader()
        #     # Create temporary config.ini file
        #     with open("config.test.ini", "w") as f:
        #         f.write("[TRAINING]\nBATCH_SIZE=32\nLEARNING_RATE=0.001\n")
        #     loader.ini_path = Path("config.test.ini")
        #     ini_vars = loader.load_ini_vars()
        #     assert ini_vars["BATCH_SIZE"] == "32"
        #     assert ini_vars["LEARNING_RATE"] == "0.001"
        #     os.remove("config.test.ini")
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
            
        # Unit test stub:
        # def test_convert_type():
        #     loader = ConfigLoader()
        #     assert loader.convert_type("42", int) == 42
        #     assert loader.convert_type("3.14", float) == 3.14
        #     assert loader.convert_type("true", bool) == True
        #     assert loader.convert_type("false", bool) == False
        #     assert loader.convert_type("hello", str) == "hello"
        """
        if value is None:
            return None
        
        # Handle string inputs - strip inline comments
        if isinstance(value, str):
            value = value.strip()
            # Strip inline comments (everything after #)
            if '#' in value:
                value = value.split('#')[0].strip()
        
        if target_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            return bool(value)
        
        if target_type == int:
            return int(float(value))  # Handle "32.0" -> 32
        
        if target_type == float:
            return float(value)
        
        if target_type == str:
            return str(value)
        
        return value
    
    def get_config_value(self, key: str, default: Any, target_type: type) -> Any:
        """
        Get configuration value with priority: ENV > INI > DEFAULT.
        
        Args:
            key: Configuration key name
            default: Default value if not found
            target_type: Target type for conversion
            
        Returns:
            Configuration value with proper type conversion
            
        # Unit test stub:
        # def test_get_config_value():
        #     loader = ConfigLoader()
        #     loader._env_vars = {"BATCH_SIZE": "64"}
        #     loader._ini_vars = {"BATCH_SIZE": "32"}
        #     # ENV should take priority
        #     assert loader.get_config_value("BATCH_SIZE", 16, int) == 64
        #     # INI should be used if ENV missing
        #     assert loader.get_config_value("LEARNING_RATE", 0.01, float) == 0.01
        """
        # Priority: Environment variables > INI file > Default
        value = None
        source = "default"
        
        if key in self._env_vars:
            value = self._env_vars[key]
            source = "environment"
        elif key in self._ini_vars:
            value = self._ini_vars[key]
            source = "ini_file"
        else:
            value = default
            self._missing_keys.append(key)
        
        try:
            converted_value = self.convert_type(value, target_type)
            logger.debug(f"Config {key}={converted_value} (from {source})")
            return converted_value
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert {key}={value} to {target_type.__name__}: {e}")
            logger.warning(f"Using default value for {key}: {default}")
            return default
    
    def create_config(self) -> Config:
        """
        Create and validate configuration by merging all sources.
        
        Returns:
            Immutable Config instance
            
        # Unit test stub:
        # def test_create_config():
        #     loader = ConfigLoader()
        #     config = loader.create_config()
        #     assert isinstance(config, Config)
        #     assert config.device in ["cuda", "cpu"]
        #     assert config.batch_size > 0
        #     assert config.learning_rate > 0
        """
        # Load from all sources
        self.load_env_vars()
        self.load_ini_vars()
        
        # Create config with type-safe value extraction
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
            
            # Model Configuration
            model_arch=self.get_config_value("MODEL_ARCH", "resnet50", str),
            use_official_processor=self.get_config_value("USE_OFFICIAL_PROCESSOR", False, bool),
            zone_focus_method=self.get_config_value("ZONE_FOCUS_METHOD", "masking", str),
            
            # Zone Masking Configuration
            use_segmentation_masking=self.get_config_value("USE_SEGMENTATION_MASKING", True, bool),
            masking_strategy=self.get_config_value("MASKING_STRATEGY", "attention", str),
            attention_strength=self.get_config_value("ATTENTION_STRENGTH", 0.7, float),
            masks_path=self.get_config_value("MASKS_PATH", "/home/pyuser/data/Paradise_Masks", str),
            
            # Internal
            _env_vars=self._env_vars.copy(),
            _ini_vars=self._ini_vars.copy(),
            _missing_keys=self._missing_keys.copy()
        )
        
        # Log missing keys
        if self._missing_keys:
            logger.warning(f"Missing configuration keys (using defaults): {', '.join(self._missing_keys)}")
        
        # Validate configuration
        self.validate_config(config)
        
        return config
    
    def validate_config(self, config: Config) -> None:
        """
        Validate configuration values and log any issues.
        
        Args:
            config: Configuration instance to validate
            
        # Unit test stub:
        # def test_validate_config():
        #     loader = ConfigLoader()
        #     config = Config(batch_size=0)  # Invalid
        #     with pytest.raises(ValueError):
        #         loader.validate_config(config)
        """
        errors = []
        
        # Validate positive integers
        if config.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {config.batch_size}")
        
        if config.n_epochs <= 0:
            errors.append(f"n_epochs must be positive, got {config.n_epochs}")
        
        if config.patience < 0:
            errors.append(f"patience must be non-negative, got {config.patience}")
        
        # Validate positive floats
        if config.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {config.learning_rate}")
        
        # Validate choices
        valid_devices = ["cuda", "cpu", "mps"]
        if config.device not in valid_devices:
            errors.append(f"device must be one of {valid_devices}, got {config.device}")
        
        valid_optimizers = ["adam", "adamw", "sgd"]
        if config.optimizer.lower() not in valid_optimizers:
            errors.append(f"optimizer must be one of {valid_optimizers}, got {config.optimizer}")
        
        # Validate zone focus method
        valid_zone_focus_methods = ["masking", "spatial_reduction"]
        if config.zone_focus_method.lower() not in valid_zone_focus_methods:
            errors.append(f"zone_focus_method must be one of {valid_zone_focus_methods}, got {config.zone_focus_method}")
        
        # Validate zone masking configuration
        valid_masking_strategies = ["zero", "attention"]
        if config.masking_strategy.lower() not in valid_masking_strategies:
            errors.append(f"masking_strategy must be one of {valid_masking_strategies}, got {config.masking_strategy}")
        
        if not (0.0 <= config.attention_strength <= 1.0):
            errors.append(f"attention_strength must be between 0.0 and 1.0, got {config.attention_strength}")
        
        # Log validation results
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            raise ValueError(f"Configuration validation failed with {len(errors)} errors")
        else:
            logger.info("Configuration validation passed")
    
    def copy_config_with_timestamp(self, config: Config) -> None:
        """
        Copy resolved config.ini to INI_DIR with timestamp when training starts.
        
        Args:
            config: Configuration instance with resolved values
        """
        from datetime import datetime
        import configparser
        from pathlib import Path
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped filename in ini_dir
        ini_dir = Path(config.ini_dir)
        ini_dir.mkdir(parents=True, exist_ok=True)
        timestamped_path = ini_dir / f"config_{timestamp}.ini"
        
        # Create new config with resolved values
        new_config = configparser.ConfigParser()
        
        # Add training section
        new_config.add_section("TRAINING")
        new_config.set("TRAINING", "BATCH_SIZE", str(config.batch_size))
        new_config.set("TRAINING", "N_EPOCHS", str(config.n_epochs))
        new_config.set("TRAINING", "PATIENCE", str(config.patience))
        new_config.set("TRAINING", "LEARNING_RATE", str(config.learning_rate))
        new_config.set("TRAINING", "OPTIMIZER", config.optimizer)
        
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
        
        # Add environment section with resolved paths
        new_config.add_section("ENVIRONMENT")
        new_config.set("ENVIRONMENT", "DEVICE", config.device)
        new_config.set("ENVIRONMENT", "DATA_SOURCE", config.data_source)
        new_config.set("ENVIRONMENT", "DATA_DIR", config.data_dir)
        new_config.set("ENVIRONMENT", "MODELS_DIR", config.models_dir)
        new_config.set("ENVIRONMENT", "CSV_DIR", config.csv_dir)
        new_config.set("ENVIRONMENT", "INI_DIR", config.ini_dir)
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
        
    # Unit test stub:
    # def test_get_config_singleton():
    #     config1 = get_config()
    #     config2 = get_config()
    #     assert config1 is config2  # Same instance
    #     
    #     config3 = get_config(force_reload=True)
    #     assert config3 is not config1  # New instance
    """
    global _config_instance
    
    if _config_instance is None or force_reload:
        logger.info("Initializing configuration...")
        loader = ConfigLoader(env_path, ini_path)
        _config_instance = loader.create_config()
        logger.info("Configuration initialized successfully")
    
    return _config_instance


def copy_config_on_training_start() -> None:
    """
    Copy resolved configuration with timestamp when training starts.
    This creates a snapshot of the actual configuration used for training.
    """
    config = get_config()
    loader = ConfigLoader()  # Don't need to pass ini_path since we're not loading from file
    loader.copy_config_with_timestamp(config)


# Singleton instance - import this in other modules
cfg = get_config()


# Export main components
__all__ = ['Config', 'ConfigLoader', 'get_config', 'copy_config_on_training_start', 'cfg']