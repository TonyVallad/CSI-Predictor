"""
General visualization utilities for CSI-Predictor.

This module contains general visualization functionality extracted from the original src/utils.py file.
"""

import torch
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from torchvision.utils import make_grid

def show_batch(
    data_loader: DataLoader,
    num_samples: int = 8,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
    config: Optional = None
) -> None:
    """
    Show a batch of images with their CSI zone predictions side-by-side.
    
    This utility randomly samples a batch and displays the 6 predicted zone masks
    in a single matplotlib figure for debugging and visualization purposes.
    
    Args:
        data_loader: DataLoader to sample from
        num_samples: Number of samples to display
        figsize: Figure size for matplotlib
        save_path: Path to save the figure (optional)
        config: Configuration object for denormalization parameters
    """
    # Import here to avoid circular imports
    if config is None:
        from src.config import cfg
        config = cfg
    
    # Get normalization parameters
    from src.data import get_normalization_parameters
    mean, std = get_normalization_parameters(config)
    
    # Get a batch from the data loader
    data_iter = iter(data_loader)
    try:
        batch_data = next(data_iter)
        # Handle both old and new data formats
        if len(batch_data) == 3:  # New format: (images, labels, file_ids)
            images, labels, file_ids = batch_data
        else:  # Old format: (images, labels)
            images, labels = batch_data
            file_ids = None
    except StopIteration:
        print("Data loader is empty")
        return
    
    # Limit to num_samples
    if images.size(0) > num_samples:
        indices = torch.randperm(images.size(0))[:num_samples]
        images = images[indices]
        labels = labels[indices]
        if file_ids is not None:
            file_ids = [file_ids[i] for i in indices]
    
    batch_size = images.size(0)
    
    # CSI zone names for labeling
    zone_names = ['Right Superior', 'Left Superior', 'Right Middle', 
                  'Left Middle', 'Right Inferior', 'Left Inferior']
    
    # CSI class names
    class_names = ['Normal (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Unknown (4)']
    
    # Create figure with subplots - very tight horizontal, more vertical space for titles
    fig, axes = plt.subplots(batch_size, 7, figsize=figsize, 
                             gridspec_kw={'wspace': 0.0, 'hspace': 0.5})
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    # Denormalize images for display using configured normalization
    mean_tensor = torch.tensor(mean).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std).view(1, 3, 1, 1)
    denorm_images = images * std_tensor + mean_tensor
    denorm_images = torch.clamp(denorm_images, 0, 1)
    
    print(f"Displaying batch with {config.normalization_strategy} denormalization")
    print(f"Mean: {mean}, Std: {std}")
    if file_ids:
        print(f"File IDs: {file_ids}")
    
    for i in range(batch_size):
        # Convert image to numpy for display
        img_np = denorm_images[i].permute(1, 2, 0).numpy()
        
        # Create title with ground truth CSI scores
        title = f'Sample {i+1}'
        if file_ids and i < len(file_ids):
            title += f'\n{file_ids[i]}'
        
        # Add ground truth CSI scores to title
        gt_scores = []
        for j in range(6):  # 6 CSI zones
            score = labels[i, j].item()
            gt_scores.append(str(score))
        title += f'\nGT: [{",".join(gt_scores)}]'
        
        # Display original image in first column
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(title, fontsize=8)
        axes[i, 0].axis('off')
        
        # Display CSI zones predictions in remaining columns
        for j, zone_name in enumerate(zone_names):
            col_idx = j + 1
            csi_score = labels[i, j].item()
            
            # Create zone visualization (simplified lung diagram)
            axes[i, col_idx].imshow(img_np)
            
            # Add zone overlay (approximate lung regions)
            h, w = img_np.shape[:2]
            zone_color = ['green', 'yellow', 'orange', 'red', 'gray'][csi_score]
            zone_alpha = 0.3
            
            # Define approximate zone regions (simplified)
            if 'Superior' in zone_name:
                y_start, y_end = 0, h//3
            elif 'Middle' in zone_name:
                y_start, y_end = h//3, 2*h//3
            else:  # Inferior
                y_start, y_end = 2*h//3, h
            
            # FIXED: Correct anatomical orientation for chest X-rays
            # Patient's RIGHT lung appears on LEFT side of image
            # Patient's LEFT lung appears on RIGHT side of image
            if 'Right' in zone_name:
                x_start, x_end = 0, w//2  # Patient's right lung = left side of image
            else:  # Left
                x_start, x_end = w//2, w  # Patient's left lung = right side of image
            
            # Add colored rectangle for the zone
            rect = patches.Rectangle(
                (x_start, y_start), x_end - x_start, y_end - y_start,
                linewidth=2, edgecolor=zone_color, facecolor=zone_color, alpha=zone_alpha
            )
            axes[i, col_idx].add_patch(rect)
            
            # Set title with zone name and score
            axes[i, col_idx].set_title(f'{zone_name}\n{class_names[csi_score]}', fontsize=8)
            axes[i, col_idx].axis('off')
    
    # Minimize horizontal spacing, more vertical space for titles
    plt.tight_layout(pad=0.0, h_pad=0.5, w_pad=0.0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.0)
        print(f"Batch visualization saved to: {save_path}")
    
    plt.show()


def visualize_data_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the distribution of CSI scores across train/val/test sets.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        save_path: Path to save the figure (optional)
    """
    # CSI zone columns
    from src.data import CSI_COLUMNS
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Unknown']
    
    for i, zone in enumerate(CSI_COLUMNS):
        # Combine data from all splits
        train_counts = train_df[zone].value_counts().sort_index()
        val_counts = val_df[zone].value_counts().sort_index()
        test_counts = test_df[zone].value_counts().sort_index()
        
        # Ensure all classes are represented
        for class_idx in range(5):
            if class_idx not in train_counts.index:
                train_counts[class_idx] = 0
            if class_idx not in val_counts.index:
                val_counts[class_idx] = 0
            if class_idx not in test_counts.index:
                test_counts[class_idx] = 0
        
        train_counts = train_counts.sort_index()
        val_counts = val_counts.sort_index()
        test_counts = test_counts.sort_index()
        
        # Create stacked bar chart
        x = np.arange(len(class_names))
        width = 0.25
        
        axes[i].bar(x - width, train_counts, width, label='Train', alpha=0.8)
        axes[i].bar(x, val_counts, width, label='Val', alpha=0.8)
        axes[i].bar(x + width, test_counts, width, label='Test', alpha=0.8)
        
        axes[i].set_title(f'{zone.replace("_", " ").title()} Zone')
        axes[i].set_xlabel('CSI Score')
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(class_names, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('CSI Score Distribution Across Train/Val/Test Sets', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    plt.show()


def analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze missing data patterns in the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with missing data statistics
    """
    from src.data import CSI_COLUMNS, CSI_UNKNOWN_CLASS
    
    analysis = {
        'total_samples': len(df),
        'zone_missing_counts': {},
        'missing_patterns': {},
        'completely_missing_samples': 0
    }
    
    # Count missing values per zone (class 4 = unknown)
    for zone in CSI_COLUMNS:
        if zone in df.columns:
            missing_count = (df[zone] == CSI_UNKNOWN_CLASS).sum()
            analysis['zone_missing_counts'][zone] = missing_count
    
    # Analyze missing patterns
    missing_matrix = df[CSI_COLUMNS] == CSI_UNKNOWN_CLASS
    patterns = missing_matrix.apply(lambda row: ''.join(row.astype(int).astype(str)), axis=1)
    analysis['missing_patterns'] = patterns.value_counts().to_dict()
    
    # Count completely missing samples
    analysis['completely_missing_samples'] = (missing_matrix.all(axis=1)).sum()
    
    return analysis

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 