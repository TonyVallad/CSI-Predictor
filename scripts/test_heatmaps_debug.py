#!/usr/bin/env python3
"""
Debug script to test heatmap generation functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Starting heatmap debug script...")

import logging
from src.config.config_loader import load_config
from src.data.dataloader import create_data_loaders
from src.models.factory import create_model
from src.evaluation.visualization.heatmaps import generate_heatmaps_for_epoch
import tempfile

print("Imports completed successfully")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_heatmap_generation():
    """Test heatmap generation functionality."""
    
    print("Loading configuration...")
    # Load configuration
    config = load_config()
    
    print("Configuration loaded successfully")
    
    # Print heatmap configuration
    print(f"Heatmap enabled: {config.heatmap_enabled}")
    print(f"Heatmap generate per epoch: {config.heatmap_generate_per_epoch}")
    print(f"Heatmap samples per epoch: {config.heatmap_samples_per_epoch}")
    
    logger.info(f"Heatmap enabled: {config.heatmap_enabled}")
    logger.info(f"Heatmap generate per epoch: {config.heatmap_generate_per_epoch}")
    logger.info(f"Heatmap samples per epoch: {config.heatmap_samples_per_epoch}")
    
    print("Creating data loaders...")
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    print("Creating model...")
    # Create model
    model = create_model(config)
    
    print("Creating temporary directory...")
    # Create temporary directory for heatmaps
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing heatmap generation in: {temp_dir}")
        logger.info(f"Testing heatmap generation in: {temp_dir}")
        
        try:
            # Test heatmap generation for epoch 1
            print("Starting heatmap generation...")
            generate_heatmaps_for_epoch(
                model=model,
                val_loader=val_loader,
                save_dir=temp_dir,
                config=config,
                epoch=1
            )
            print("Heatmap generation test completed successfully!")
            logger.info("Heatmap generation test completed successfully!")
            
            # List generated files
            import glob
            heatmap_files = glob.glob(os.path.join(temp_dir, "*.png"))
            print(f"Generated {len(heatmap_files)} heatmap files:")
            logger.info(f"Generated {len(heatmap_files)} heatmap files:")
            for file in heatmap_files:
                print(f"  - {os.path.basename(file)}")
                logger.info(f"  - {os.path.basename(file)}")
                
        except Exception as e:
            print(f"Heatmap generation test failed: {e}")
            logger.error(f"Heatmap generation test failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    print("Starting main function...")
    test_heatmap_generation()
    print("Script completed.")
