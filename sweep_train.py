#!/usr/bin/env python3
"""
Dedicated sweep training script following W&B best practices.
This script is designed to be called by wandb sweep agent.
"""

import os
import sys
import wandb
from pathlib import Path
from dataclasses import replace

# Disable legacy service and improve stability
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DISABLE_ARTIFACT'] = 'true'
os.environ['WANDB_REQUIRE_SERVICE'] = 'false'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import cfg, get_config
from src.training.trainer import train_model

def main():
    """
    Main training function for W&B sweep.
    This follows the official W&B sweep pattern.
    """
    # Get the base configuration
    base_config = get_config()
    
    # Set wandb directory environment variable globally
    # This prevents wandb from creating a .wandb folder in the current directory
    os.environ['WANDB_DIR'] = base_config.wandb_dir
    
    # Ensure the wandb directory exists
    os.makedirs(base_config.wandb_dir, exist_ok=True)
    
    print(f"Using wandb directory: {base_config.wandb_dir}")
    
    # Initialize wandb - this is crucial for sweep functionality
    # Note: wandb will create a 'wandb' subfolder inside our specified directory
    # This is normal behavior, so our files will be in /path/to/wandb/wandb/
    with wandb.init(dir=base_config.wandb_dir) as run:
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