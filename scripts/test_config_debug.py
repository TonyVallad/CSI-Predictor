#!/usr/bin/env python3
"""
Simple script to test configuration loading for heatmap settings.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing configuration loading...")

try:
    # Try to import and load config
    from src.config.config_loader import load_config
    
    config = load_config()
    
    print("Configuration loaded successfully!")
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
    print(f"Error loading configuration: {e}")
    import traceback
    traceback.print_exc()
