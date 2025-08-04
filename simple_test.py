#!/usr/bin/env python3
"""
Very simple test script to verify wandb sweep functionality.
"""

import os
import wandb

# Disable legacy service
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLE_ARTIFACT'] = 'true'
os.environ['WANDB_REQUIRE_SERVICE'] = 'false'

def main():
    """Simple test function that just logs a metric."""
    
    # Initialize wandb
    with wandb.init() as run:
        print(f"Wandb run initialized: {run.id}")
        print(f"Wandb config: {dict(run.config)}")
        
        # Simulate a metric value
        metric_value = 0.5
        
        # Log the metric that the sweep is optimizing
        wandb.log({'val_f1_weighted': metric_value})
        
        print(f"Logged val_f1_weighted: {metric_value}")
        print("Test completed successfully!")

if __name__ == "__main__":
    main() 