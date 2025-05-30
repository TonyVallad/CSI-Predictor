"""
Pure PyTorch data splitting utilities for CSI-Predictor.
Replaces scikit-learn's train_test_split with native implementations.
"""

import torch
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from collections import Counter


def create_stratification_groups(df: pd.DataFrame, strat_columns: List[str]) -> pd.Series:
    """
    Create stratification groups based on specified columns.
    
    Args:
        df: DataFrame with data
        strat_columns: Column names to use for stratification
        
    Returns:
        Series with group identifiers for each row
    """
    # Create binary mask for each stratification column (e.g., unknown values)
    group_keys = []
    for col in strat_columns:
        if col in df.columns:
            # Create binary indicator (e.g., 1 if unknown/4, 0 otherwise)
            binary_indicator = (df[col] == 4).astype(int)
            group_keys.append(binary_indicator.astype(str))
    
    if group_keys:
        # Combine all binary indicators into a single group key
        combined_key = pd.concat(group_keys, axis=1).apply(
            lambda row: ''.join(row), axis=1
        )
        return combined_key
    else:
        # Fallback: single group
        return pd.Series(['0'] * len(df), index=df.index)


def stratified_split_indices(
    indices: np.ndarray, 
    stratify_by: np.ndarray, 
    test_size: float, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform stratified split of indices.
    
    Args:
        indices: Array of indices to split
        stratify_by: Array of stratification labels (same length as indices)
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_indices, test_indices)
    """
    np.random.seed(random_state)
    
    # Get unique groups and their counts
    unique_groups, group_counts = np.unique(stratify_by, return_counts=True)
    
    train_indices = []
    test_indices = []
    
    for group, count in zip(unique_groups, group_counts):
        # Get indices for this group
        group_mask = stratify_by == group
        group_indices = indices[group_mask]
        
        # Calculate split sizes
        n_test = max(1, int(count * test_size))  # At least 1 sample for test
        n_train = count - n_test
        
        # Shuffle indices for this group
        shuffled_indices = np.random.permutation(group_indices)
        
        # Split
        train_indices.extend(shuffled_indices[:n_train])
        test_indices.extend(shuffled_indices[n_train:n_train + n_test])
    
    return np.array(train_indices), np.array(test_indices)


def pytorch_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    stratify_by: Optional[pd.Series] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test sets using PyTorch/NumPy.
    
    Args:
        df: DataFrame to split
        test_size: Fraction for test set
        stratify_by: Series for stratification (optional)
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    np.random.seed(random_state)
    
    # Get all indices
    all_indices = np.arange(len(df))
    
    if stratify_by is not None:
        # Check if stratification is feasible
        group_counts = Counter(stratify_by)
        min_group_size = min(group_counts.values())
        
        if min_group_size < 2:
            print(f"Warning: Smallest group has only {min_group_size} sample(s). Using random split.")
            stratify_by = None
    
    if stratify_by is not None:
        # Stratified split
        train_indices, test_indices = stratified_split_indices(
            all_indices, stratify_by.values, test_size, random_state
        )
    else:
        # Random split
        shuffled_indices = np.random.permutation(all_indices)
        n_test = int(len(df) * test_size)
        
        test_indices = shuffled_indices[:n_test]
        train_indices = shuffled_indices[n_test:]
    
    # Create DataFrames
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    
    return train_df, test_df


def pytorch_train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify_columns: Optional[List[str]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets using PyTorch.
    
    Args:
        df: DataFrame to split
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        stratify_columns: Columns to use for stratification
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Validate sizes
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Train, val, and test sizes must sum to 1.0")
    
    print(f"Splitting data: train={train_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")
    
    # Create stratification groups if specified
    stratify_by = None
    if stratify_columns:
        stratify_by = create_stratification_groups(df, stratify_columns)
        n_groups = len(stratify_by.unique())
        print(f"Created {n_groups} unique stratification groups")
    
    # First split: train vs (val + test)
    temp_size = val_size + test_size
    train_df, temp_df = pytorch_train_test_split(
        df, 
        test_size=temp_size, 
        stratify_by=stratify_by,
        random_state=random_state
    )
    
    # Second split: val vs test from temp_df
    # Recalculate stratification for remaining data
    if stratify_by is not None:
        temp_stratify = create_stratification_groups(temp_df, stratify_columns)
    else:
        temp_stratify = None
    
    # Calculate proportions for val/test split
    val_proportion = val_size / temp_size
    
    val_df, test_df = pytorch_train_test_split(
        temp_df,
        test_size=1 - val_proportion,
        stratify_by=temp_stratify,
        random_state=random_state + 1  # Different seed for second split
    )
    
    print(f"Split completed: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


# Export main functions
__all__ = [
    'pytorch_train_test_split',
    'pytorch_train_val_test_split',
    'create_stratification_groups',
    'stratified_split_indices'
] 