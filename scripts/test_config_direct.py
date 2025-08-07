#!/usr/bin/env python3
"""
Direct script to test configuration loading for heatmap settings.
This script reads config.ini directly without importing complex modules.
"""

import configparser
import os
from pathlib import Path

print("Testing configuration loading directly from config.ini...")

try:
    # Read config.ini directly
    config = configparser.ConfigParser()
    config_path = "config/config.ini"
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found at: {config_path}")
        # Try alternative paths
        alt_paths = ["../config/config.ini", "../../config/config.ini"]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config_path = alt_path
                print(f"✓ Found config file at: {config_path}")
                break
        else:
            print("✗ Could not find config.ini file")
            exit(1)
    
    config.read(config_path)
    
    # Check if HEATMAP section exists
    if 'HEATMAP' not in config:
        print("✗ HEATMAP section not found in config.ini")
        print("Available sections:", list(config.sections()))
        exit(1)
    
    # Read heatmap settings
    heatmap_enabled = config.getboolean('HEATMAP', 'HEATMAP_ENABLED', fallback=True)
    heatmap_generate_per_epoch = config.getboolean('HEATMAP', 'HEATMAP_GENERATE_PER_EPOCH', fallback=False)
    heatmap_samples_per_epoch = config.getint('HEATMAP', 'HEATMAP_SAMPLES_PER_EPOCH', fallback=1)
    heatmap_color_map = config.get('HEATMAP', 'HEATMAP_COLOR_MAP', fallback='custom_purple_red')
    
    print("✓ Configuration loaded successfully!")
    print(f"Heatmap enabled: {heatmap_enabled}")
    print(f"Heatmap generate per epoch: {heatmap_generate_per_epoch}")
    print(f"Heatmap samples per epoch: {heatmap_samples_per_epoch}")
    print(f"Heatmap color map: {heatmap_color_map}")
    
    # Test the condition that should trigger per-epoch generation
    condition = heatmap_enabled and heatmap_generate_per_epoch
    print(f"Condition (heatmap_enabled AND heatmap_generate_per_epoch): {condition}")
    
    if condition:
        print("✓ Per-epoch heatmap generation should work!")
    else:
        print("✗ Per-epoch heatmap generation will be skipped")
        if not heatmap_enabled:
            print("  - Reason: heatmap_enabled is False")
        if not heatmap_generate_per_epoch:
            print("  - Reason: heatmap_generate_per_epoch is False")
    
    # Show the actual config.ini content for debugging
    print("\n--- Config.ini HEATMAP section ---")
    for key, value in config['HEATMAP'].items():
        print(f"  {key} = {value}")
            
except Exception as e:
    print(f"✗ Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
