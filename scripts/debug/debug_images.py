#!/usr/bin/env python3
"""
Debug script to visualize NIFTI images and check orientation.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.dataloader import create_data_loaders
from src.utils.visualization import show_batch
from src.config import cfg


def main():
    """Create data loaders and visualize sample images for debugging."""
    print("Creating data loaders...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(cfg)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Get a batch from train loader
    print("\nGetting a batch from train loader...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image dtype: {images.dtype}")
        print(f"  Labels dtype: {labels.dtype}")
        print(f"  Image min/max: {images.min():.3f}/{images.max():.3f}")
        print(f"  Labels min/max: {labels.min()}/{labels.max()}")
        
        # Show the batch
        show_batch(images, labels, title=f"Batch {batch_idx}")
        
        if batch_idx >= 2:  # Show only first 3 batches
            break
    
    print("\nDebug visualization completed!")


if __name__ == "__main__":
    main() 