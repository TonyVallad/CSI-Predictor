#!/usr/bin/env python3
"""
Debug script to visualize NIFTI images and check orientation.
Run this after switching to ImageNet normalization to verify images look correct.
"""

import sys
import os
from pathlib import Path
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
    
    # Create debug directory
    debug_dir = Path(getattr(cfg, 'debug_dir', './debug_output'))
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Debug directory: {debug_dir}")
    
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
    
    # Show 10 training images with very tight spacing
    try:
        train_output_path = debug_dir / "debug_train_batch.png"
        show_batch(
            data_loader=train_loader, 
            num_samples=10, 
            figsize=(20, 15),  # Much smaller figure for very tight spacing
            save_path=str(train_output_path),
            config=cfg
        )
        print(f"✅ Training batch visualization complete!")
    except Exception as e:
        print(f"❌ Failed to visualize training batch: {e}")
        return
    
    # Show 10 validation images
    try:
        val_output_path = debug_dir / "debug_val_batch.png"
        show_batch(
            data_loader=val_loader, 
            num_samples=10, 
            figsize=(20, 15),  # Much smaller figure for very tight spacing
            save_path=str(val_output_path),
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
    print(f"   7. ✅ Ground truth (GT) scores should make sense for visible pathology")
    
    print(f"\n📸 Output files:")
    print(f"   - {train_output_path}")
    print(f"   - {val_output_path}")
    
    print(f"\n📊 Ground Truth Legend:")
    print(f"   GT: [R_Sup, L_Sup, R_Mid, L_Mid, R_Inf, L_Inf]")
    print(f"   Scores: 0=Normal, 1=Mild, 2=Moderate, 3=Severe, 4=Unknown")
    
    print(f"\n🔧 If images look wrong:")
    print(f"   1. Check coordinate corrections in _load_nifti_image()")
    print(f"   2. Try different combinations of transpose/flip")
    print(f"   3. Compare with original DICOM images if available")
    print(f"   4. Verify that ground truth scores match visible pathology")

if __name__ == "__main__":
    main() 