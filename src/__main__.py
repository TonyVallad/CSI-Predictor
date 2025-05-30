"""
CLI entry point for CSI-Predictor src module.
Supports python -m src.train and python -m src.evaluate commands.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src <command> [args]")
        print("Commands:")
        print("  train    - Train CSI-Predictor model")
        print("  evaluate - Evaluate CSI-Predictor model")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        # Remove the command from sys.argv so train script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from train import main
        main()
    elif command == "evaluate":
        # Remove the command from sys.argv so evaluate script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from evaluate import main
        main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, evaluate")
        sys.exit(1) 