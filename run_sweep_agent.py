#!/usr/bin/env python3
"""
Wrapper script to run wandb sweep agent with proper environment variables.
This prevents wandb from creating folders in the project root.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import config to get the wandb directory
from src.config import get_config

def main():
    """Run the sweep agent with proper environment variables."""
    
    # Get the configuration
    config = get_config()
    
    # Set environment variables
    env = os.environ.copy()
    env['WANDB_DIR'] = config.wandb_dir
    env['WANDB_SILENT'] = 'true'
    env['WANDB_DISABLE_ARTIFACT'] = 'true'
    env['WANDB_REQUIRE_SERVICE'] = 'false'
    
    print(f"Setting WANDB_DIR to: {config.wandb_dir}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Get the sweep ID from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_sweep_agent.py <sweep_id>")
        print("Example: python run_sweep_agent.py tony-vallad-chru-de-nancy/csi-predictor/8p0nptfv")
        sys.exit(1)
    
    sweep_id = sys.argv[1]
    
    # Run the wandb agent with the proper environment
    cmd = ["wandb", "agent", sweep_id]
    
    print(f"Running: {' '.join(cmd)}")
    print(f"With WANDB_DIR: {env['WANDB_DIR']}")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Sweep agent failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nSweep agent interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main() 