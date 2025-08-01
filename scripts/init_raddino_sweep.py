#!/usr/bin/env python3
"""
Initialize W&B sweep for RadDINO model only.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from modules to avoid package __init__.py issues
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from optimization.wandb_sweep import initialize_model_sweep
from config import get_config

def main():
    """Initialize sweep for RadDINO only."""
    
    # Load configuration
    config = get_config()
    
    print("🚀 Initializing W&B Sweep for RadDINO")
    print("=" * 50)
    
    try:
        sweep_id = initialize_model_sweep(
            project_name="csi-predictor",
            model_arch="raddino"
        )
        print(f"✅ RadDINO sweep created: {sweep_id}")
        print(f"🔗 URL: https://wandb.ai/csi-predictor/sweeps/{sweep_id}")
        print(f"🤖 Run agent with: wandb agent csi-predictor/csi-predictor/{sweep_id}")
    except Exception as e:
        print(f"❌ Failed to create RadDINO sweep: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 RadDINO sweep initialization complete!")

if __name__ == "__main__":
    main() 