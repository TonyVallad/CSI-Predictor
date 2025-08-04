#!/usr/bin/env python3
"""
Dedicated sweep training script following W&B best practices.
This script is designed to be called by wandb sweep agent.
"""

import os
import sys
from pathlib import Path

# Set environment variables BEFORE ANY imports
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLE_ARTIFACT'] = 'true'
os.environ['WANDB_REQUIRE_SERVICE'] = 'false'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import config to get the wandb directory
from src.config import cfg, get_config

# Set wandb directory environment variable BEFORE importing wandb
base_config = get_config()
os.environ['WANDB_DIR'] = base_config.wandb_dir

# Now import wandb after setting environment variables
import wandb
from dataclasses import replace
from src.training.trainer import train_model

def main():
    """
    Main training function for W&B sweep.
    This follows the official W&B sweep pattern.
    """
    # Ensure the wandb directory exists
    os.makedirs(base_config.wandb_dir, exist_ok=True)
    
    print(f"Using wandb directory: {base_config.wandb_dir}")
    
    # Initialize wandb - this is crucial for sweep functionality
    wandb_parent_dir = os.path.dirname(base_config.wandb_dir)
    print(f"Wandb parent directory: {wandb_parent_dir}")
    
    with wandb.init(dir=wandb_parent_dir) as run:
        print(f"Wandb run initialized: {run.id}")
        print(f"Wandb config: {dict(run.config)}")
        print(f"Wandb directory: {base_config.wandb_dir}")
        
        # Create a mutable copy of the configuration with sweep hyperparameters
        config_updates = {}
        for key, value in run.config.items():
            if hasattr(base_config, key):
                config_updates[key] = value
                print(f"Will update config.{key} = {value}")
        
        print(f"Configuration updates: {config_updates}")
        
        # Create a new mutable configuration with the sweep parameters
        mutable_config = replace(base_config, **config_updates)
        
        # Run training with the mutable configuration
        try:
            train_model(mutable_config)
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed: {e}")
            # Log error to wandb
            wandb.log({'error': str(e)})
            raise

if __name__ == "__main__":
    main() 