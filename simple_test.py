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
        
        # Simulate a metric value based on hyperparameters
        # This makes the test more realistic
        learning_rate = run.config.get('learning_rate', 0.001)
        batch_size = run.config.get('batch_size', 32)
        
        # Create a fake metric that depends on the hyperparameters
        metric_value = 0.5 + (learning_rate * 100) + (batch_size / 1000)
        
        # Log the metric that the sweep is optimizing
        wandb.log({'val_f1_weighted': metric_value})
        
        print(f"Logged val_f1_weighted: {metric_value}")
        print("Test completed successfully!")

if __name__ == "__main__":
    main() 