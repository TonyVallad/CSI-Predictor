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

# Import nibabel for NIFTI file support
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None

from .config import Config, cfg
from .data_split import pytorch_train_val_test_split

# Try to import transformers for RadDINO processor
try:
    from transformers import AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoImageProcessor = None

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
    'chexnet': (224, 224),
    'custom1': (224, 224),
    'raddino': (518, 518),  # RadDINO's expected input size from Microsoft
}

# CSI zone column names (6 zones)
CSI_COLUMNS = ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']

# CSI class mapping: 0-3 are actual scores, 4 is unknown/ungradable
CSI_UNKNOWN_CLASS = 4


def get_normalization_parameters(config: Optional[Config] = None) -> Tuple[List[float], List[float]]:
    """
    Get normalization mean and std based on the configured strategy.
    
    Args:
        config: Configuration object (uses global cfg if None)
        
    Returns:
        Tuple of (mean, std) lists for normalization
    """
    if config is None:
        config = cfg
    
    strategy = config.normalization_strategy.lower()
    
    if strategy == "imagenet":
        # Standard ImageNet normalization
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif strategy == "medical":
        # Medical image normalization (assumes grayscale converted to 3-channel)
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    elif strategy == "simple":
        # Simple normalization (no mean/std subtraction, just 0-1 range)
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    elif strategy == "custom":
        # Custom normalization values
        if config.custom_mean is None or config.custom_std is None:
            print("Warning: Custom normalization strategy selected but custom_mean/custom_std not provided. Falling back to medical.")
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        return config.custom_mean, config.custom_std
    else:
        print(f"Warning: Unknown normalization strategy '{strategy}'. Falling back to medical.")
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def get_raddino_processor(use_official: bool = False):
    """
    Get RadDINO image processor.
    
    Args:
        use_official: Whether to use the official AutoImageProcessor
        
    Returns:
        AutoImageProcessor or None
    """
    if use_official and TRANSFORMERS_AVAILABLE:
        try:
            repo = "microsoft/rad-dino"
            processor = AutoImageProcessor.from_pretrained(repo, use_fast=True)
            return processor
        except Exception as e:
            print(f"Warning: Failed to load RadDINO processor: {e}")
            return None
    return None


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
        
        # Apply FileID exclusion filter
        if cfg.excluded_file_ids:
            initial_count = len(df)
            # Convert FileID to string for comparison (handles both string and numeric FileIDs)
            df = df[~df['FileID'].astype(str).isin(cfg.excluded_file_ids)]
            excluded_count = initial_count - len(df)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} FileIDs based on exclusion filter")
                print(f"Excluded FileIDs: {', '.join(cfg.excluded_file_ids)}")
                print(f"Remaining samples: {len(df)}")
            else:
                print(f"No FileIDs were excluded (none found in dataset)")
        else:
            print("No FileID exclusion filter configured")
        
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


def get_default_transforms(phase: str = "train", model_arch: str = "resnet50", use_official_processor: bool = False, config: Optional[Config] = None) -> transforms.Compose:
    """
    Get default image transformations for different phases.
    
    Args:
        phase: Phase name ('train', 'val', 'test')
        model_arch: Model architecture name for input size
        use_official_processor: Whether to use official RadDINO processor (RadDINO only)
        config: Configuration object for normalization parameters
        
    Returns:
        Composed transformations
    """
    # Get input size for model architecture
    input_size = MODEL_INPUT_SIZES.get(model_arch, (224, 224))
    
    # Get normalization parameters
    mean, std = get_normalization_parameters(config)
    
    # Special handling for RadDINO with official processor
    if (model_arch.lower().replace('_', '').replace('-', '') == 'raddino' and 
        use_official_processor):
        # Return minimal transforms - the official processor will handle everything
        return transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor (processor expects PIL images)
        ])
    
    # Special handling for RadDINO which has its own preprocessing
    if model_arch.lower().replace('_', '').replace('-', '') == 'raddino':
        if phase == "train":
            return transforms.Compose([
                transforms.Resize(input_size),
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                # RadDINO uses ImageNet normalization
                transforms.Normalize(mean=mean, std=std)
            ])
        else:  # val or test
            return transforms.Compose([
                transforms.Resize(input_size),
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
                transforms.ToTensor(),
                # RadDINO uses ImageNet normalization
                transforms.Normalize(mean=mean, std=std)
            ])
    
    # Standard transforms for other models
    if phase == "train":
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
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
        load_to_memory: bool = False,
        use_official_processor: bool = False,
        model_arch: str = "resnet50",
        config: Optional[Config] = None
    ):
        """
        Initialize CSI dataset.
        
        Args:
            dataframe: DataFrame with FileID and CSI zone columns
            data_path: Path to image directory
            transform: Image transformations (optional)
            phase: Dataset phase for default transforms
            load_to_memory: Whether to pre-cache images in memory
            use_official_processor: Whether to use official RadDINO processor
            model_arch: Model architecture for default transforms
            config: Configuration object for normalization and file format settings
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_path = Path(data_path)
        self.phase = phase
        self.load_to_memory = load_to_memory
        self.use_official_processor = use_official_processor
        self.model_arch = model_arch
        self.config = config if config is not None else cfg
        
        # Set up processor for RadDINO if needed
        self.raddino_processor = None
        if (self.model_arch.lower().replace('_', '').replace('-', '') == 'raddino' and 
            self.use_official_processor):
            self.raddino_processor = get_raddino_processor(use_official=True)
            if self.raddino_processor is None:
                print("Warning: Failed to load RadDINO processor, falling back to standard transforms")
                self.use_official_processor = False
        
        # Set default transform if none provided
        if transform is None:
            self.transform = get_default_transforms(phase, model_arch, use_official_processor, self.config)
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
        if self.use_official_processor and self.raddino_processor:
            print(f"Using official RadDINO AutoImageProcessor")
        if self.load_to_memory:
            print(f"Pre-cached {len(self.cached_images)} images in memory")
    
    def _load_nifti_image(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Load and preprocess a NIFTI image with coordinate corrections.
        
        Args:
            image_path: Path to the NIFTI file
            
        Returns:
            Preprocessed image array as float32, or None if loading failed
        """
        if not NIBABEL_AVAILABLE:
            raise RuntimeError("nibabel is required for NIFTI support. Install with: pip install nibabel")
        
        try:
            # Load NIFTI file
            nifti_img = nib.load(str(image_path))
            img_data = nifti_img.get_fdata().astype(np.float32)
            
            # Handle potential 3D NIFTI with single slice (squeeze any singleton dimensions)
            img_data = np.squeeze(img_data)
            
            # Ensure we have 2D data
            if len(img_data.shape) > 2:
                print(f"Warning: Unexpected NIFTI shape {img_data.shape} for {image_path}")
                return None
            
            # Apply coordinate corrections (same as functions_image_exploration.py)
            # Fix NIFTI orientation (transpose to correct counterclockwise rotation, then flip horizontally)
            img_data = np.transpose(img_data)
            img_data = np.fliplr(img_data)  # Flip left-right to correct horizontal mirroring
            
            # Normalize to 0-1 range (NIFTI files already have 99th percentile clipping applied)
            if img_data.max() > img_data.min():
                img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            else:
                img_data = np.zeros_like(img_data, dtype=np.float32)
            
            return img_data
            
        except Exception as e:
            print(f"Error loading NIFTI file {image_path}: {e}")
            return None
    
    def _cache_images(self) -> None:
        """Pre-cache all images in memory."""
        print(f"Pre-caching {len(self.dataframe)} images in memory...")
        
        failed_files = []
        
        for idx in tqdm(range(len(self.dataframe)), desc="Caching images"):
            file_id = self.dataframe.iloc[idx]['FileID']
            # Use NIFTI extension from config
            image_filename = f"{file_id}{self.config.image_extension}"
            image_path = self.data_path / image_filename
            
            try:
                # Check if file exists first
                if not image_path.exists():
                    print(f"Warning: NIFTI file not found: {image_path}")
                    failed_files.append(f"{file_id} (file not found)")
                    self.cached_images[idx] = None
                    continue
                
                # Load NIFTI file as float32 array
                image_array = self._load_nifti_image(image_path)
                if image_array is not None:
                    # Store the raw float32 array (coordinate-corrected and normalized to 0-1)
                    self.cached_images[idx] = image_array
                else:
                    # Store None for failed images
                    print(f"Warning: Failed to load NIFTI data from {image_path}")
                    failed_files.append(f"{file_id} (load failed)")
                    self.cached_images[idx] = None
            except Exception as e:
                print(f"Warning: Failed to cache image {image_path}: {e}")
                failed_files.append(f"{file_id} (exception: {e})")
                # Store None for failed images
                self.cached_images[idx] = None
        
        # Report summary
        successful_count = len(self.dataframe) - len(failed_files)
        print(f"Caching complete: {successful_count}/{len(self.dataframe)} images loaded successfully")
        
        if failed_files:
            print(f"Failed to load {len(failed_files)} images:")
            for failed_file in failed_files[:10]:  # Show first 10 failures
                print(f"  - {failed_file}")
            if len(failed_files) > 10:
                print(f"  ... and {len(failed_files) - 10} more")
            
            # Suggest solutions
            print("\nPossible solutions:")
            print("1. Check that all FileIDs in your CSV have corresponding .nii.gz files")
            print("2. Verify the data path is correct:", self.data_path)
            print("3. Ensure NIFTI files were created successfully during conversion")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataframe)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get dataset item with NIFTI loading support.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, label_tensor, file_id)
            
        Note: Returns classification targets (class indices) for cross-entropy loss.
        Label tensor shape: (6,) with values in {0,1,2,3,4} representing classes.
        """
        row = self.dataframe.iloc[idx]
        file_id = str(row['FileID'])  # Ensure file_id is string for consistency
        
        # Load image (from cache or disk)
        if self.load_to_memory and idx in self.cached_images:
            image_array = self.cached_images[idx]
            if image_array is None:
                # Handle cached failed image with more specific error
                image_filename = f"{file_id}{self.config.image_extension}"
                image_path = self.data_path / image_filename
                raise RuntimeError(f"Failed to load cached NIFTI image for FileID '{file_id}' (index {idx}). "
                                 f"Expected file: {image_path}. "
                                 f"Check that this NIFTI file exists and is valid.")
        else:
            # Load from disk
            image_filename = f"{file_id}{self.config.image_extension}"
            image_path = self.data_path / image_filename
            try:
                image_array = self._load_nifti_image(image_path)
                if image_array is None:
                    raise RuntimeError(f"Failed to load NIFTI image for FileID '{file_id}': {image_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load image for FileID '{file_id}' from {image_path}: {e}")
        
        # Convert single-channel float32 array to PIL Image for transform compatibility
        # Scale to 0-255 range and convert to uint8
        image_uint8 = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL Image (grayscale)
        image = Image.fromarray(image_uint8, mode='L')
        
        # Apply preprocessing
        if (self.use_official_processor and self.raddino_processor and 
            self.model_arch.lower().replace('_', '').replace('-', '') == 'raddino'):
            # Use official RadDINO processor
            try:
                inputs = self.raddino_processor(images=image, return_tensors="pt")
                image = inputs['pixel_values'].squeeze(0)  # Remove batch dimension
            except Exception as e:
                print(f"Warning: RadDINO processor failed for image {idx}: {e}")
                # Fall back to standard transforms
                if self.transform:
                    image = self.transform(image)
        else:
            # Use standard transforms
            if self.transform:
                image = self.transform(image)
        
        # Extract CSI labels (6 zones) as classification targets
        labels = []
        for col in CSI_COLUMNS:
            labels.append(row[col])
        
        # Convert to classification targets (already in range 0-4)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Return file_id for zone masking support
        return image, label_tensor, file_id


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
        print("Loading and splitting data...")
        train_df, val_df, test_df = load_and_split_data()
    
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