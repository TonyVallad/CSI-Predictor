"""
Data Pipeline for CSI-Predictor.

This module provides comprehensive data handling functionality including:
- CSV ingestion with specific column loading
- NaN conversion to "unknown/ungradable" class (index 4)
- Stratified train/val/test splitting based on unknown score presence
- Torch-compatible Dataset with lazy/cached image loading
- Configurable transforms with model-specific input sizes
- Batch visualization utilities

Usage:
    from src.data import create_data_loaders, load_and_split_data
    
    # Load and split data
    train_df, val_df, test_df = load_and_split_data()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders()
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .config import Config, cfg
from .data_split import pytorch_train_val_test_split


# Model architecture to input size mapping
MODEL_INPUT_SIZES = {
    'resnet18': (224, 224),
    'resnet34': (224, 224),
    'resnet50': (224, 224),
    'resnet101': (224, 224),
    'resnet152': (224, 224),
    'efficientnet_b0': (224, 224),
    'efficientnet_b1': (240, 240),
    'efficientnet_b2': (260, 260),
    'efficientnet_b3': (300, 300),
    'efficientnet_b4': (380, 380),
    'densenet121': (224, 224),
    'densenet169': (224, 224),
    'densenet201': (224, 224),
    'vit_base_patch16_224': (224, 224),
    'vit_large_patch16_224': (224, 224),
}

# CSI zone column names (6 zones)
CSI_COLUMNS = ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']

# CSI class mapping: 0-3 are actual scores, 4 is unknown/ungradable
CSI_UNKNOWN_CLASS = 4


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data with specific columns required for CSI prediction.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with FileID and CSI zone columns
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading CSV data from {csv_path}")
    
    # Load only required columns
    required_columns = ['FileID'] + CSI_COLUMNS
    
    try:
        # Use the CSV separator from configuration
        df = pd.read_csv(csv_path, usecols=required_columns, sep=cfg.labels_csv_separator)
        print(f"Loaded {len(df)} samples from CSV")
        
        # Check for missing columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV data: {e}")


def convert_nans_to_unknown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert NaN values in CSI zone columns to unknown class (index 4).
    
    Args:
        df: DataFrame with CSI zone columns
        
    Returns:
        DataFrame with NaNs converted to unknown class
    """
    df_processed = df.copy()
    
    # Convert NaN values to unknown class (4) in CSI columns
    for col in CSI_COLUMNS:
        if col in df_processed.columns:
            nan_count = df_processed[col].isna().sum()
            if nan_count > 0:
                print(f"Converting {nan_count} NaN values to unknown class in column '{col}'")
                df_processed[col] = df_processed[col].fillna(CSI_UNKNOWN_CLASS)
            
            # Ensure integer type
            df_processed[col] = df_processed[col].astype(int)
    
    return df_processed


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """
    Create stratification key based on presence of unknown scores.
    
    Args:
        df: DataFrame with CSI zone columns
        
    Returns:
        Series with stratification keys
    """
    # Create binary mask for unknown values (class 4) in each zone
    unknown_mask = []
    for col in CSI_COLUMNS:
        if col in df.columns:
            unknown_mask.append((df[col] == CSI_UNKNOWN_CLASS).astype(int))
    
    # Convert to stratification key (binary string representation)
    if unknown_mask:
        unknown_matrix = pd.concat(unknown_mask, axis=1)
        # Create unique stratification keys
        strat_keys = unknown_matrix.apply(lambda row: ''.join(row.astype(str)), axis=1)
        return strat_keys
    else:
        # Fallback: use first CSI column for stratification
        return df[CSI_COLUMNS[0]] if CSI_COLUMNS[0] in df.columns else pd.Series([0] * len(df))


def split_data_stratified(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test with stratification over unknown score presence.
    
    Args:
        df: DataFrame to split
        train_size: Fraction for training set
        val_size: Fraction for validation set  
        test_size: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Train, val, and test sizes must sum to 1.0")
    
    print(f"Splitting data: train={train_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")
    
    # Use PyTorch implementation with CSI columns for stratification
    try:
        train_df, val_df, test_df = pytorch_train_val_test_split(
            df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            stratify_columns=CSI_COLUMNS,  # Stratify based on CSI columns
            random_state=random_state
        )
        
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"Warning: Stratified split failed ({e}), using random split")
        
        # Fallback to random split using our PyTorch implementation
        from .data_split import pytorch_train_test_split
        
        # First split: train vs (val + test)
        train_df, temp_df = pytorch_train_test_split(
            df,
            test_size=val_size + test_size,
            stratify_by=None,  # No stratification for fallback
            random_state=random_state
        )
        
        # Second split: val vs test
        val_prop = val_size / (val_size + test_size)
        val_df, test_df = pytorch_train_test_split(
            temp_df,
            test_size=1 - val_prop,
            stratify_by=None,
            random_state=random_state + 1
        )
        
        print(f"Split completed: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df


def get_default_transforms(phase: str = "train", model_arch: str = "resnet50") -> transforms.Compose:
    """
    Get default image transformations for different phases.
    
    Args:
        phase: Phase name ('train', 'val', 'test')
        model_arch: Model architecture name for input size
        
    Returns:
        Composed transformations
    """
    # Get input size for model architecture
    input_size = MODEL_INPUT_SIZES.get(model_arch, (224, 224))
    
    if phase == "train":
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class CSIDataset(Dataset):
    """
    Torch-compatible Dataset for CSI prediction.
    
    Supports both lazy loading from disk and pre-caching in memory based on configuration.
    Returns image tensors and label tensors where labels are shape (6,) with values in {0,1,2,3,4}.
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_path: str,
        transform: Optional[transforms.Compose] = None,
        phase: str = "train",
        load_to_memory: bool = False
    ):
        """
        Initialize CSI dataset.
        
        Args:
            dataframe: DataFrame with FileID and CSI zone columns
            data_path: Path to image directory
            transform: Image transformations (optional)
            phase: Dataset phase for default transforms
            load_to_memory: Whether to pre-cache images in memory
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.phase = phase
        self.load_to_memory = load_to_memory
        
        # Set default transform if none provided
        if transform is None:
            self.transform = get_default_transforms(phase, cfg.model_arch)
        else:
            self.transform = transform
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # Pre-cache images if requested
        self.cached_images = {}
        if self.load_to_memory:
            self._cache_images()
        
        print(f"Initialized {phase} dataset with {len(self.dataframe)} samples")
        if self.load_to_memory:
            print(f"Pre-cached {len(self.cached_images)} images in memory")
    
    def _cache_images(self) -> None:
        """Pre-cache all images in memory."""
        print(f"Pre-caching {len(self.dataframe)} images in memory...")
        
        for idx in tqdm(range(len(self.dataframe)), desc="Caching images"):
            file_id = self.dataframe.iloc[idx]['FileID']
            # Automatically append .png extension to FileID
            image_filename = f"{file_id}.png"
            image_path = self.data_path / image_filename
            
            try:
                # Load and store raw PIL image (before transforms)
                image = Image.open(image_path).convert('RGB')
                self.cached_images[idx] = image
            except Exception as e:
                print(f"Warning: Failed to cache image {image_path}: {e}")
                # Store None for failed images
                self.cached_images[idx] = None
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
            
        Note: Returns classification targets (class indices) for cross-entropy loss.
        Label tensor shape: (6,) with values in {0,1,2,3,4} representing classes.
        """
        row = self.dataframe.iloc[idx]
        file_id = row['FileID']
        
        # Load image (from cache or disk)
        if self.load_to_memory and idx in self.cached_images:
            image = self.cached_images[idx]
            if image is None:
                # Handle cached failed image
                raise RuntimeError(f"Cached image at index {idx} is None")
        else:
            # Load from disk
            # Automatically append .png extension to FileID
            image_filename = f"{file_id}.png"
            image_path = self.data_path / image_filename
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Extract CSI labels (6 zones) as classification targets
        labels = []
        for col in CSI_COLUMNS:
            labels.append(row[col])
        
        # Convert to classification targets (already in range 0-4)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        return image, label_tensor


def load_and_split_data(
    csv_path: Optional[str] = None,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV data and split into train/val/test sets.
    
    Args:
        csv_path: Path to CSV file (uses cfg.csv_path if None)
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if csv_path is None:
        csv_path = cfg.csv_path
    
    # Load and process CSV data
    df = load_csv_data(csv_path)
    df = convert_nans_to_unknown(df)
    
    # Split data
    train_df, val_df, test_df = split_data_stratified(
        df, train_size, val_size, test_size, random_state
    )
    
    return train_df, val_df, test_df


def create_data_loaders(
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    config: Optional[Config] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
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
    
    # Load and split data if DataFrames not provided
    if train_df is None or val_df is None or test_df is None:
        print("Loading and splitting data...")
        train_df, val_df, test_df = load_and_split_data()
    
    # Create datasets
    train_dataset = CSIDataset(
        dataframe=train_df,
        data_path=config.data_path,
        transform=get_default_transforms("train", config.model_arch),
        phase="train",
        load_to_memory=config.load_data_to_memory
    )
    
    val_dataset = CSIDataset(
        dataframe=val_df,
        data_path=config.data_path,
        transform=get_default_transforms("val", config.model_arch),
        phase="val",
        load_to_memory=config.load_data_to_memory
    )
    
    test_dataset = CSIDataset(
        dataframe=test_df,
        data_path=config.data_path,
        transform=get_default_transforms("test", config.model_arch),
        phase="test",
        load_to_memory=config.load_data_to_memory
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # Re-enabled for better performance (was 0)
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,  # Re-enabled for better performance (was 0)
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,  # Re-enabled for better performance (was 0)
        pin_memory=True
    )
    
    print(f"Created data loaders: train={len(train_loader)} batches, "
          f"val={len(val_loader)} batches, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader 