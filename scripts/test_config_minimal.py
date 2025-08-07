#!/usr/bin/env python3
"""
Minimal script to test configuration loading for heatmap settings.
This script avoids complex dependencies like loguru.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("Testing configuration loading...")
print(f"Project root: {project_root}")

try:
    # Try to import and load config without using the logging module
    from src.config.config import Config
    from src.config.config_loader import ConfigLoader
    
    # Create a simple config loader
    loader = ConfigLoader(".env", "config/config.ini")
    config = loader.create_config()
    
    print("✓ Configuration loaded successfully!")
    print(f"Heatmap enabled: {config.heatmap_enabled}")
    print(f"Heatmap generate per epoch: {config.heatmap_generate_per_epoch}")
    print(f"Heatmap samples per epoch: {config.heatmap_samples_per_epoch}")
    print(f"Heatmap color map: {config.heatmap_color_map}")
    
    # Test the condition that should trigger per-epoch generation
    condition = config.heatmap_enabled and config.heatmap_generate_per_epoch
    print(f"Condition (heatmap_enabled AND heatmap_generate_per_epoch): {condition}")
    
    if condition:
        print("✓ Per-epoch heatmap generation should work!")
    else:
        print("✗ Per-epoch heatmap generation will be skipped")
        if not config.heatmap_enabled:
            print("  - Reason: heatmap_enabled is False")
        if not config.heatmap_generate_per_epoch:
            print("  - Reason: heatmap_generate_per_epoch is False")
            
except Exception as e:
    print(f"✗ Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
