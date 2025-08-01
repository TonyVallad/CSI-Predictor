"""
Data preprocessing utilities for CSI-Predictor.

This module contains data preprocessing functionality extracted from the original src/data.py file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from ..config import Config, cfg, ANSI

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
            print(f"{ANSI['Y']}Warning: Custom normalization strategy selected but custom_mean/custom_std not provided. Falling back to medical.{ANSI['W']}")
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        return config.custom_mean, config.custom_std
    else:
        print(f"{ANSI['Y']}Warning: Unknown normalization strategy '{strategy}'. Falling back to medical.{ANSI['W']}")
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV data with specific columns required for CSI prediction.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with FileID, CSI zone columns, and CSI average column
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"{ANSI['Y']}Loading CSV data from:{ANSI['W']} {csv_path}")
    
    # Load required columns plus CSI average column
    required_columns = ['FileID'] + CSI_COLUMNS + ['csi']
    
    try:
        # Use the CSV separator from configuration
        df = pd.read_csv(csv_path, usecols=required_columns, sep=cfg.labels_csv_separator)
        print(f"{ANSI['B']}Loaded{ANSI['W']} {len(df)} samples from CSV")
        
        # Check for missing columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            print(f"{ANSI['Y']}Warning: Missing columns in CSV:{ANSI['W']} {missing_cols}")
            # If 'csi' column is missing, calculate it from individual zones
            if 'csi' in missing_cols:
                print(f"{ANSI['B']}Calculating CSI average from individual zone columns...{ANSI['W']}")
                df['csi'] = df[CSI_COLUMNS].mean(axis=1)
                print(f"{ANSI['G']}CSI average column calculated successfully{ANSI['W']}")
            else:
                raise ValueError(f"Missing required columns in CSV: {missing_cols}")
        
        # Apply FileID exclusion filter
        if cfg.excluded_file_ids:
            initial_count = len(df)
            # Convert FileID to string for comparison (handles both string and numeric FileIDs)
            df = df[~df['FileID'].astype(str).isin(cfg.excluded_file_ids)]
            excluded_count = initial_count - len(df)
            if excluded_count > 0:
                print(f"{ANSI['Y']}Excluded{ANSI['W']} {excluded_count} FileIDs based on exclusion filter")
                print(f"{ANSI['Y']}Excluded FileIDs:{ANSI['W']} {', '.join(cfg.excluded_file_ids)}")
                print(f"{ANSI['B']}Remaining samples:{ANSI['W']} {len(df)}")
            else:
                print(f"{ANSI['B']}No FileIDs were excluded (none found in dataset){ANSI['W']}")
        else:
            print(f"{ANSI['B']}No FileID exclusion filter configured{ANSI['W']}")
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV data: {e}")


def filter_existing_files(df: pd.DataFrame, data_path: str, image_extension: str = '.nii.gz') -> pd.DataFrame:
    """
    Filter DataFrame to include FileIDs that have corresponding image files.
    
    Args:
        df: DataFrame with FileID column
        data_path: Path to image directory
        image_extension: Extension for image files
        
    Returns:
        Filtered DataFrame with only existing files
    """
    data_path = Path(data_path)
    missing_files = []
    existing_files = []
    
    print(f"{ANSI['Y']}Checking for existing image files in:{ANSI['W']} {data_path}")
    print(f"{ANSI['B']}Looking for files with extension:{ANSI['W']} {image_extension}")
    
    for idx, row in df.iterrows():
        file_id = row['FileID']
        # Convert FileID to integer to remove .0 suffix
        try:
            file_id_int = int(float(file_id))
            image_filename = f"{file_id_int}{image_extension}"
        except (ValueError, TypeError):
            # If conversion fails, use original file_id
            image_filename = f"{file_id}{image_extension}"
        
        image_path = data_path / image_filename
        
        if image_path.exists():
            existing_files.append(idx)
        else:
            missing_files.append(file_id)
    
    # Filter DataFrame to only include existing files
    filtered_df = df.loc[existing_files].reset_index(drop=True)
    
    print(f"{ANSI['B']}File existence check complete:{ANSI['W']}")
    print(f"  - Total files in CSV: {len(df)}")
    print(f"  - Existing files: {len(filtered_df)}")
    print(f"  - Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"{ANSI['Y']}Missing files (first 10):{ANSI['W']} {missing_files[:10]}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return filtered_df


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

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 