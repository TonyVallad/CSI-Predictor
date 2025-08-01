#!/usr/bin/env python3
"""
Initialize W&B sweep for ResNet50 model only.
Simple standalone version that creates sweep directly.
"""

import sys
import os
from pathlib import Path
import wandb

def get_resnet50_sweep_config():
    """Get W&B sweep configuration for ResNet50."""
    return {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'val_f1_weighted',
            'goal': 'maximize'
        },
        'parameters': {
            # Fixed model parameters
            'model_arch': {
                'value': 'resnet50'
            },
            'use_official_processor': {
                'value': False
            },
            'batch_size': {
                'value': 128
            },
            
            # Sweep parameters
            'optimizer': {
                'values': ['adam', 'adamw', 'sgd']
            },
            'learning_rate': {
                'distribution': 'log_uniform_values',
                'min': 0.0001,
                'max': 0.01
            },
            'weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 0.000001,
                'max': 0.001
            },
            'dropout_rate': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.7
            },
            'momentum': {
                'distribution': 'uniform',
                'min': 0.8,
                'max': 0.99
            },
            'normalization_strategy': {
                'values': ['imagenet', 'medical']
            },
            'scheduler_type': {
                'values': ['ReduceLROnPlateau', 'CosineAnnealingLR']
            },
            'unknown_weight': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 1.0
            },
            'patience': {
                'value': 15
            }
        },
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 5,
            'eta': 3
        }
    }

def main():
    """Initialize sweep for ResNet50 only."""
    
    print("🚀 Initializing W&B Sweep for ResNet50")
    print("=" * 50)
    
    try:
        # Get sweep configuration
        sweep_config = get_resnet50_sweep_config()
        sweep_config['name'] = "CSI-Predictor ResNet50 Hyperparameter Optimization"
        
        # Initialize sweep
        sweep_id = wandb.sweep(sweep_config, project="csi-predictor")
        
        print(f"✅ ResNet50 sweep created: {sweep_id}")
        print(f"🔗 URL: https://wandb.ai/tony-vallad-chru-de-nancy/csi-predictor/sweeps/{sweep_id}")
        print(f"🤖 Run agent with: wandb agent tony-vallad-chru-de-nancy/csi-predictor/{sweep_id}")
        
    except Exception as e:
        print(f"❌ Failed to create ResNet50 sweep: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 ResNet50 sweep initialization complete!")

if __name__ == "__main__":
    main() 