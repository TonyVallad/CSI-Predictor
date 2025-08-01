#!/usr/bin/env python3
"""
Initialize W&B sweep for CheXNet model only.
Standalone version that bypasses package imports.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from the module file to avoid package __init__.py issues
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the specific functions we need
import importlib.util

# Load wandb_sweep module directly
wandb_sweep_path = Path(__file__).parent.parent / "src" / "optimization" / "wandb_sweep.py"
spec = importlib.util.spec_from_file_location("wandb_sweep", wandb_sweep_path)
wandb_sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wandb_sweep)

# Load config module directly
config_path = Path(__file__).parent.parent / "src" / "config" / "__init__.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

def main():
    """Initialize sweep for CheXNet only."""
    
    # Load configuration
    config_instance = config.get_config()
    
    print("🚀 Initializing W&B Sweep for CheXNet")
    print("=" * 50)
    
    try:
        sweep_id = wandb_sweep.initialize_model_sweep(
            project_name="csi-predictor",
            model_arch="chexnet"
        )
        print(f"✅ CheXNet sweep created: {sweep_id}")
        print(f"🔗 URL: https://wandb.ai/csi-predictor/sweeps/{sweep_id}")
        print(f"🤖 Run agent with: wandb agent csi-predictor/csi-predictor/{sweep_id}")
    except Exception as e:
        print(f"❌ Failed to create CheXNet sweep: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🎉 CheXNet sweep initialization complete!")

if __name__ == "__main__":
    main() 