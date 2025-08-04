#!/usr/bin/env python3
"""
Simple test script to verify wandb sweep functionality.
"""

import os
import sys
import wandb
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_sweep_function():
    """Test function that logs the metric the sweep is optimizing."""
    
    # Initialize wandb
    with wandb.init() as run:
        print(f"Wandb run initialized: {run.id}")
        print(f"Wandb config: {dict(run.config)}")
        
        # Simulate training and compute a fake metric
        # This should match the metric name in the sweep config
        fake_val_f1_weighted = 0.5  # Simulate a metric value
        
        # Log the metric that the sweep is optimizing
        wandb.log({'val_f1_weighted': fake_val_f1_weighted})
        
        print(f"Logged val_f1_weighted: {fake_val_f1_weighted}")
        print("Test completed successfully!")

if __name__ == "__main__":
    test_sweep_function() 