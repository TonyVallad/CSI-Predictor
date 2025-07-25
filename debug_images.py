#!/usr/bin/env python3
"""
Debug script to visualize NIFTI images and check orientation.
Run this after switching to ImageNet normalization to verify images look correct.
"""

import sys
import os
sys.path.append('src')

from src.data import create_data_loaders
from src.utils import show_batch
from src.config import cfg

def main():
    print("🔍 NIFTI Image Debugging Script")
    print("=" * 50)
    
    # Display current configuration
    print(f"📊 Current Configuration:")
    print(f"   Data path: {cfg.data_path}")
    print(f"   Image format: {cfg.image_format}")
    print(f"   Image extension: {cfg.image_extension}")
    print(f"   Normalization strategy: {cfg.normalization_strategy}")
    print(f"   Device: {cfg.device}")
    print(f"   Load to memory: {cfg.load_data_to_memory}")
    
    print(f"\n🏗️  Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders()
        print(f"✅ Data loaders created successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")
        return
    
    print(f"\n🖼️  Visualizing sample images...")
    
    # Show a batch from training data
    try:
        show_batch(
            data_loader=train_loader, 
            num_samples=4, 
            figsize=(20, 10),
            save_path="debug_train_batch.png",
            config=cfg
        )
        print(f"✅ Training batch visualization complete!")
    except Exception as e:
        print(f"❌ Failed to visualize training batch: {e}")
        return
    
    # Show a batch from validation data
    try:
        show_batch(
            data_loader=val_loader, 
            num_samples=2, 
            figsize=(15, 6),
            save_path="debug_val_batch.png",
            config=cfg
        )
        print(f"✅ Validation batch visualization complete!")
    except Exception as e:
        print(f"❌ Failed to visualize validation batch: {e}")
        return
    
    print(f"\n🎯 What to check in the visualized images:")
    print(f"   1. ✅ Images should look like normal chest X-rays (not rotated/flipped)")
    print(f"   2. ✅ Patient's RIGHT lung should appear on LEFT side of image")
    print(f"   3. ✅ Patient's LEFT lung should appear on RIGHT side of image")
    print(f"   4. ✅ Heart should be visible on the left side of the image")
    print(f"   5. ✅ Spine should be visible in the center")
    print(f"   6. ✅ Zone overlays should match anatomical regions correctly")
    
    print(f"\n📸 Output files:")
    print(f"   - debug_train_batch.png")
    print(f"   - debug_val_batch.png")
    
    print(f"\n🔧 If images look wrong:")
    print(f"   1. Check coordinate corrections in _load_nifti_image()")
    print(f"   2. Try different combinations of transpose/flip")
    print(f"   3. Compare with original DICOM images if available")

if __name__ == "__main__":
    main() 