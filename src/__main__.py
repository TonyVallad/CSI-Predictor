"""
CLI entry point for CSI-Predictor src module.
Supports python -m src.train and python -m src.evaluate commands.
"""

import sys
from pathlib import Path
from .config import ANSI

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{ANSI['B']}Usage:{ANSI['W']} python -m src <command> [args]")
        print(f"{ANSI['B']}Commands:{ANSI['W']}")
        print(f"  {ANSI['G']}train{ANSI['W']}    - Train CSI-Predictor model")
        print(f"  {ANSI['G']}evaluate{ANSI['W']} - Evaluate CSI-Predictor model")
        print(f"  {ANSI['G']}optimize{ANSI['W']} - Run hyperparameter optimization")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        # Remove the command from sys.argv so train script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .cli.train import train_cli
        from .cli.train import create_train_parser
        parser = create_train_parser()
        args = parser.parse_args()
        train_cli(args)
    elif command == "evaluate":
        # Remove the command from sys.argv so evaluate script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .cli.evaluate import evaluate_cli
        from .cli.evaluate import create_evaluate_parser
        parser = create_evaluate_parser()
        args = parser.parse_args()
        evaluate_cli(args)
    elif command == "optimize":
        # Remove the command from sys.argv so optimize script gets correct args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .cli.optimize import optimize_cli
        from .cli.optimize import create_optimize_parser
        parser = create_optimize_parser()
        args = parser.parse_args()
        optimize_cli(args)
    else:
        print(f"{ANSI['R']}Unknown command:{ANSI['W']} {command}")
        print(f"{ANSI['B']}Available commands:{ANSI['W']} train, evaluate, optimize")
        sys.exit(1) 