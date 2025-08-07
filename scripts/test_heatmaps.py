#!/usr/bin/env python3
"""
Test script for heatmap generation functionality.

This script tests the heatmap generation without requiring a full training run.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config.config_loader import load_config
from src.data.dataloader import create_data_loaders
from src.models.factory import create_model
from src.evaluation.visualization.heatmaps import (
    generate_heatmaps_for_best_model,
    generate_heatmaps_for_epoch,
    create_custom_purple_red_colormap
)
from src.utils.logging import logger

def test_heatmap_generation():
    """Test heatmap generation functionality."""
    
    logger.info("Testing heatmap generation functionality...")
    
    # Load configuration
    try:
        test_config = load_config()
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return False
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = create_data_loaders(test_config)
        logger.info("✓ Data loaders created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create data loaders: {e}")
        return False
    
    # Create a simple test model
    try:
        model = create_model(test_config)
        logger.info("✓ Model created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create model: {e}")
        return False
    
    # Test colormap creation
    try:
        colormap = create_custom_purple_red_colormap()
        logger.info("✓ Custom colormap created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create colormap: {e}")
        return False
    
    # Create test output directory
    test_output_dir = Path("test_heatmaps")
    test_output_dir.mkdir(exist_ok=True)
    
    # Test heatmap generation
    try:
        generate_heatmaps_for_best_model(
            model=model,
            val_loader=val_loader,
            save_dir=str(test_output_dir),
            config=test_config
        )
        logger.info("✓ Heatmap generation completed successfully")
        
        # Check if files were created
        heatmap_files = list(test_output_dir.glob("heatmap_*.png"))
        if heatmap_files:
            logger.info(f"✓ Generated {len(heatmap_files)} heatmap files")
            for file in heatmap_files:
                logger.info(f"  - {file.name}")
        else:
            logger.warning("⚠ No heatmap files were generated")
            
    except Exception as e:
        logger.error(f"✗ Failed to generate heatmaps: {e}")
        return False
    
    logger.info("✓ All heatmap tests passed!")
    return True

if __name__ == "__main__":
    success = test_heatmap_generation()
    sys.exit(0 if success else 1)
