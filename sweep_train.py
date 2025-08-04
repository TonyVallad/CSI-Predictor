#!/usr/bin/env python3
"""
Dedicated sweep training script following W&B best practices.
This script is designed to be called by wandb sweep agent.
"""

import os
import sys
import wandb
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import cfg
from src.training.trainer import train_model

def main():
    """
    Main training function for W&B sweep.
    This follows the official W&B sweep pattern.
    """
    # Initialize wandb - this is crucial for sweep functionality
    with wandb.init() as run:
        print(f"Wandb run initialized: {run.id}")
        print(f"Wandb config: {dict(run.config)}")
        
        # Update configuration with sweep hyperparameters
        config_updates = {}
        for key, value in run.config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                config_updates[key] = value
                print(f"Updated config.{key} = {value}")
        
        print(f"Configuration updated: {config_updates}")
        
        # Run training
        try:
            train_model(cfg)
            print("Training completed successfully!")
        except Exception as e:
            print(f"Training failed: {e}")
            # Log error to wandb
            wandb.log({'error': str(e)})
            raise

if __name__ == "__main__":
    main() 