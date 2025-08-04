#!/bin/bash
# Shell script to run wandb sweep agent with proper environment variables
# This prevents wandb from creating folders in the project root

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add src to Python path
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# Run Python to get the wandb directory from config
WANDB_DIR=$(python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/src')
from src.config import get_config
config = get_config()
print(config.wandb_dir)
")

# Set environment variables
export WANDB_DIR="$WANDB_DIR"
export WANDB_SILENT="true"
export WANDB_DISABLE_ARTIFACT="true"
export WANDB_REQUIRE_SERVICE="false"

echo "Setting WANDB_DIR to: $WANDB_DIR"
echo "Current working directory: $(pwd)"

# Check if sweep ID is provided
if [ $# -eq 0 ]; then
    echo "Usage: ./run_sweep.sh <sweep_id>"
    echo "Example: ./run_sweep.sh tony-vallad-chru-de-nancy/csi-predictor/8p0nptfv"
    exit 1
fi

SWEEP_ID="$1"

echo "Running: wandb agent $SWEEP_ID"
echo "With WANDB_DIR: $WANDB_DIR"

# Run the wandb agent
wandb agent "$SWEEP_ID" 