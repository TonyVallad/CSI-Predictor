#!/usr/bin/env python3
"""
Test script to verify wandb logging functionality for CSI-Predictor sweeps.
This script tests if the val_f1_weighted metric can be logged correctly to wandb.
"""

import wandb
import torch
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import cfg
from src.utils.logging import logger

def test_wandb_logging():
    """Test wandb logging with a simple sweep-like configuration."""
    
    # Initialize wandb with a test project
    try:
        wandb.init(
            project="csi-predictor-test",
            name="test-wandb-logging",
            dir=cfg.wandb_dir,
            config={
                "test_param": 0.5,
                "learning_rate": 0.001,
                "batch_size": 32
            }
        )
        logger.info("Wandb initialized successfully")
        
        # Simulate training epochs with fake metrics
        for epoch in range(5):
            # Generate fake metrics
            val_f1_weighted = 0.3 + (epoch * 0.1) + np.random.normal(0, 0.05)
            train_f1_weighted = 0.4 + (epoch * 0.1) + np.random.normal(0, 0.05)
            val_loss = 1.0 - (epoch * 0.1) + np.random.normal(0, 0.05)
            train_loss = 0.8 - (epoch * 0.1) + np.random.normal(0, 0.05)
            
            # Ensure metrics are valid
            val_f1_weighted = max(0.0, min(1.0, val_f1_weighted))
            train_f1_weighted = max(0.0, min(1.0, train_f1_weighted))
            val_loss = max(0.0, val_loss)
            train_loss = max(0.0, train_loss)
            
            # Log metrics
            log_dict = {
                'epoch': epoch,
                'val_f1_weighted': val_f1_weighted,
                'train_f1_weighted': train_f1_weighted,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'learning_rate': 0.001
            }
            
            try:
                wandb.log(log_dict)
                logger.info(f"Epoch {epoch}: Successfully logged val_f1_weighted = {val_f1_weighted}")
            except Exception as e:
                logger.error(f"Failed to log metrics for epoch {epoch}: {e}")
                logger.error(f"Log dict: {log_dict}")
        
        # Log final metric
        final_val_f1 = 0.75  # Simulate a good final score
        try:
            wandb.log({'val_f1_weighted': final_val_f1})
            logger.info(f"Final val_f1_weighted logged successfully: {final_val_f1}")
        except Exception as e:
            logger.error(f"Failed to log final metric: {e}")
        
        # Finish the run
        wandb.finish()
        logger.info("Wandb test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in wandb test: {e}")
        raise

def test_sweep_like_logging():
    """Test wandb logging in a sweep-like context."""
    
    # Create a simple sweep configuration
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_f1_weighted',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 0.01
            },
            'batch_size': {
                'values': [16, 32, 64]
            }
        }
    }
    
    try:
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project="csi-predictor-test")
        logger.info(f"Created sweep with ID: {sweep_id}")
        
        # Define training function
        def train_function():
            with wandb.init(dir=cfg.wandb_dir) as run:
                logger.info(f"Wandb run initialized: {run.id}")
                
                # Get config from wandb
                config = wandb.config
                logger.info(f"Wandb config: {dict(config)}")
                
                # Simulate training
                for epoch in range(3):
                    val_f1_weighted = 0.3 + (epoch * 0.2) + np.random.normal(0, 0.05)
                    val_f1_weighted = max(0.0, min(1.0, val_f1_weighted))
                    
                    log_dict = {
                        'epoch': epoch,
                        'val_f1_weighted': val_f1_weighted,
                        'val_loss': 1.0 - (epoch * 0.2)
                    }
                    
                    wandb.log(log_dict)
                    logger.info(f"Sweep epoch {epoch}: val_f1_weighted = {val_f1_weighted}")
                
                # Log final metric
                final_val_f1 = 0.8
                wandb.log({'val_f1_weighted': final_val_f1})
                logger.info(f"Sweep final val_f1_weighted: {final_val_f1}")
        
        # Run the agent
        wandb.agent(sweep_id, train_function, count=2)
        logger.info("Sweep test completed successfully")
        
    except Exception as e:
        logger.error(f"Error in sweep test: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting wandb logging tests...")
    
    try:
        # Test basic wandb logging
        test_wandb_logging()
        
        # Test sweep-like logging
        test_sweep_like_logging()
        
        logger.info("All wandb tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1) 