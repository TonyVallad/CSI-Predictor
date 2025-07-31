"""
Main configuration class for CSI-Predictor.

This module contains the main configuration functionality extracted from the original src/config.py file.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        data_dir: Master data directory (parent for all data subdirectories)
        nifti_dir: Directory containing NIFTI images
        models_dir: Directory for saving/loading models
        csv_dir: Directory containing CSV files
        ini_dir: Directory containing config files
        png_dir: Directory for PNG images
        graph_dir: Directory for saving graphs and visualizations
        debug_dir: Directory for debug visualizations
        masks_dir: Directory for segmentation masks
        logs_dir: Directory for saving logs and training history
        runs_dir: Directory for training runs
        evaluation_dir: Directory for evaluation outputs
        wandb_dir: Directory for Weights & Biases files
        
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
    data_dir: str = "./data"  # Master data directory (defaults to ./data if not set in .env)
    nifti_dir: str = ""  # Directory containing NIFTI images (resolved dynamically)
    dicom_dir: str = ""  # Directory containing DICOM images (resolved dynamically)
    models_dir: str = ""  # Directory for saving/loading models (resolved dynamically)
    csv_dir: str = ""  # Directory containing CSV files (resolved dynamically)
    ini_dir: str = ""  # Directory containing config files (resolved dynamically)
    png_dir: str = ""  # Directory for PNG images (resolved dynamically)
    graph_dir: str = ""  # Directory for saving graphs and visualizations (resolved dynamically)
    debug_dir: str = ""  # Directory for debug visualizations (resolved dynamically)
    masks_dir: str = ""  # Directory for segmentation masks (resolved dynamically)
    logs_dir: str = ""  # Directory for saving logs and training history (resolved dynamically)
    runs_dir: str = ""  # Directory for training runs (resolved dynamically)
    evaluation_dir: str = ""  # Directory for evaluation outputs (resolved dynamically)
    wandb_dir: str = ""  # Directory for Weights & Biases files (resolved dynamically)
    
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
        """Legacy property for backward compatibility - returns nifti_dir."""
        return self.nifti_dir
    
    @property
    def csv_path(self) -> str:
        """Construct full path to labels CSV file."""
        return os.path.join(self.csv_dir, self.labels_csv)
    
    @property
    def models_folder(self) -> str:
        """Legacy property for backward compatibility."""
        return self.models_dir
    
    @property
    def model_path(self) -> str:
        """Get path to the final trained model."""
        return os.path.join(self.models_dir, "final_model.pth")
    
    @property
    def masks_path(self) -> str:
        """Legacy property for backward compatibility - returns masks_dir."""
        return self.masks_dir
    
    def get_model_path(self, model_name: str, extension: str = "pth") -> str:
        """
        Get full path for a model file.
        
        Args:
            model_name: Name of the model (without extension)
            extension: File extension (default: pth)
            
        Returns:
            Full path to the model file
        """
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Create model filename
        if not model_name.endswith(f".{extension}"):
            model_name = f"{model_name}.{extension}"
        
        return os.path.join(self.models_dir, model_name)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 