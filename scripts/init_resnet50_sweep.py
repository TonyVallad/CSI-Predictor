#!/usr/bin/env python3
"""
Initialize W&B sweep for ResNet50 model only.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization.wandb_sweep import initialize_model_sweep
from src.config import get_config

def main():
    """Initialize sweep for ResNet50 only."""
    
    # Load configuration
    config = get_config()
    
    print("🚀 Initializing W&B Sweep for ResNet50")
    print("=" * 50)
    
    try:
        sweep_id = initialize_model_sweep(
            project_name="csi-predictor",
            model_arch="resnet50"
        )
        print(f"✅ ResNet50 sweep created: {sweep_id}")
        print(f"🔗 URL: https://wandb.ai/csi-predictor/sweeps/{sweep_id}")
        print(f"🤖 Run agent with: wandb agent csi-predictor/csi-predictor/{sweep_id}")
    except Exception as e:
        print(f"❌ Failed to create ResNet50 sweep: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 ResNet50 sweep initialization complete!")

if __name__ == "__main__":
    main() 