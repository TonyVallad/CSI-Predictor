#!/usr/bin/env python3
"""
Initialize W&B sweeps for CSI-Predictor models.

This script creates sweeps for ResNet50, CheXNet, and RadDINO models.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.wandb_sweep import initialize_model_sweep
from src.config import get_config

def main():
    """Initialize sweeps for all model architectures."""
    
    # Load configuration
    config = get_config()
    
    # Initialize sweeps for each model
    models = ['resnet50', 'chexnet', 'raddino']
    
    print("🚀 Initializing W&B Sweeps for CSI-Predictor")
    print("=" * 50)
    
    for model_arch in models:
        print(f"\n📊 Creating sweep for {model_arch.upper()}...")
        try:
            sweep_id = initialize_model_sweep(
                project_name="csi-predictor",
                model_arch=model_arch
            )
            print(f"✅ {model_arch.upper()} sweep created: {sweep_id}")
            print(f"🔗 URL: https://wandb.ai/csi-predictor/sweeps/{sweep_id}")
            print(f"🤖 Run agent with: wandb agent csi-predictor/csi-predictor/{sweep_id}")
        except Exception as e:
            print(f"❌ Failed to create {model_arch.upper()} sweep: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Sweep initialization complete!")
    print("\nNext steps:")
    print("1. Visit your W&B dashboard to see the sweeps")
    print("2. Run agents for each sweep:")
    print("   wandb agent csi-predictor/csi-predictor/<SWEEP_ID>")
    print("3. Or run multiple agents in parallel for faster optimization")

if __name__ == "__main__":
    main() 