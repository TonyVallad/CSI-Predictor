"""
Data loader creation for CSI-Predictor.

This module contains data loader functionality extracted from the original src/data.py file.
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from ..config import Config, cfg, ANSI
from .preprocessing import load_csv_data, convert_nans_to_unknown, filter_existing_files
from .splitting import split_data_stratified
from .transforms import get_default_transforms
from .dataset import CSIDataset

def load_and_split_data(
    csv_path: Optional[str] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    data_path: Optional[str] = None,
    image_extension: Optional[str] = None
) -> Tuple:
    """
    Load CSV data and split into train/val/test sets.
    
    Args:
        csv_path: Path to CSV file (uses cfg.csv_path if None)
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_state: Random seed
        data_path: Path to image directory (uses cfg.data_path if None)
        image_extension: Extension for image files (uses cfg.image_extension if None)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if csv_path is None:
        csv_path = cfg.csv_path
    if data_path is None:
        data_path = cfg.data_path
    if image_extension is None:
        image_extension = cfg.image_extension
    
    # Load and process CSV data
    df = load_csv_data(csv_path)
    df = convert_nans_to_unknown(df)
    
    # Filter to only include files that actually exist
    df = filter_existing_files(df, data_path, image_extension)
    
    # Split data
    train_df, val_df, test_df = split_data_stratified(
        df, train_size, val_size, test_size, random_state
    )
    
    return train_df, val_df, test_df


def create_data_loaders(
    train_df: Optional = None,
    val_df: Optional = None,
    test_df: Optional = None,
    config: Optional[Config] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing with optional RadDINO processor support.
    
    Args:
        train_df: Training DataFrame (loads from CSV if None)
        val_df: Validation DataFrame (loads from CSV if None)
        test_df: Test DataFrame (loads from CSV if None)
        config: Configuration object (uses global cfg if None)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = cfg
    
    # Get processor setting for RadDINO
    use_official_processor = getattr(config, 'use_official_processor', False)
    
    # Load and split data if DataFrames not provided
    if train_df is None or val_df is None or test_df is None:
        print(f"{ANSI['B']}Loading and splitting data...{ANSI['W']}")
        train_df, val_df, test_df = load_and_split_data(
            data_path=config.data_path,
            image_extension=config.image_extension
        )
    
    # Create datasets with processor option
    train_dataset = CSIDataset(
        dataframe=train_df,
        data_path=config.data_path,
        transform=get_default_transforms("train", config.model_arch, use_official_processor, config),
        phase="train",
        load_to_memory=config.load_data_to_memory,
        use_official_processor=use_official_processor,
        model_arch=config.model_arch,
        config=config
    )
    
    val_dataset = CSIDataset(
        dataframe=val_df,
        data_path=config.data_path,
        transform=get_default_transforms("val", config.model_arch, use_official_processor, config),
        phase="val",
        load_to_memory=config.load_data_to_memory,
        use_official_processor=use_official_processor,
        model_arch=config.model_arch,
        config=config
    )
    
    test_dataset = CSIDataset(
        dataframe=test_df,
        data_path=config.data_path,
        transform=get_default_transforms("test", config.model_arch, use_official_processor, config),
        phase="test",
        load_to_memory=config.load_data_to_memory,
        use_official_processor=use_official_processor,
        model_arch=config.model_arch,
        config=config
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for container environments to avoid shared memory issues
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for container environments to avoid shared memory issues
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for container environments to avoid shared memory issues
        pin_memory=True
    )
    
    print(f"Created data loaders: train={len(train_loader)} batches, "
          f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 