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
        print("  optimize - Run hyperparameter optimization")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        # Remove the command from sys.argv so train script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from cli.train import train_cli
        from cli.main import create_train_parser
        parser = create_train_parser()
        args = parser.parse_args()
        train_cli(args)
    elif command == "evaluate":
        # Remove the command from sys.argv so evaluate script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from cli.evaluate import evaluate_cli
        from cli.main import create_evaluate_parser
        parser = create_evaluate_parser()
        args = parser.parse_args()
        evaluate_cli(args)
    elif command == "optimize":
        # Remove the command from sys.argv so optimize script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from cli.optimize import optimize_cli
        from cli.main import create_optimize_parser
        parser = create_optimize_parser()
        args = parser.parse_args()
        optimize_cli(args)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, evaluate, optimize")
        sys.exit(1) 