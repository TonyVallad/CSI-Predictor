"""
Data splitting utilities for CSI-Predictor.

This module contains data splitting functionality extracted from the original src/data.py file.
"""

import pandas as pd
from typing import Tuple
from .data_split import pytorch_train_val_test_split, pytorch_train_test_split
from ..config import ANSI

# CSI zone column names (6 zones)
CSI_COLUMNS = ['right_sup', 'left_sup', 'right_mid', 'left_mid', 'right_inf', 'left_inf']

# CSI class mapping: 0-3 are actual scores, 4 is unknown/ungradable
CSI_UNKNOWN_CLASS = 4

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
    
    print(f"{ANSI['B']}Splitting data:{ANSI['W']} train={train_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")
    
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
        print(f"{ANSI['Y']}Warning: Stratified split failed ({e}), using random split{ANSI['W']}")
        
        # Fallback to random split using our PyTorch implementation
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
        
        print(f"{ANSI['G']}Split completed:{ANSI['W']} train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 