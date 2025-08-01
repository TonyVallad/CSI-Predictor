"""
Optimization CLI for CSI-Predictor.

This module contains optimization CLI functionality extracted from the original main.py file.
"""

import argparse
from loguru import logger

from src.optimization.hyperopt import optimize_hyperparameters
from src.optimization.wandb_sweep import initialize_sweep, run_sweep_agent
from src.config import ANSI

def optimize_cli(args, mode="hyperopt"):
    """
    Handle optimization CLI commands.
    
    Args:
        args: Parsed command line arguments
        mode: Optimization mode ("hyperopt", "sweep", or "sweep-agent")
    """
    if mode == "hyperopt":
        logger.info("Starting hyperparameter optimization...")
        
        # Handle WandB project configuration
        wandb_project = args.wandb_project if args.wandb_project.lower() != 'none' else None
        
        if wandb_project:
            logger.info(f"WandB logging enabled - Project: {wandb_project}")
            logger.info("You can monitor progress in real-time at: https://wandb.ai/")
        else:
            logger.info("WandB logging disabled")
        
        study = optimize_hyperparameters(
            study_name=args.study_name,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs,
            config_path=args.config,
            sampler=args.sampler,
            pruner=args.pruner,
            wandb_project=wandb_project
        )
        logger.info("Hyperparameter optimization completed.")
        
        # Print recommended next steps
        print(f"\n{ANSI['B']}{'='*80}{ANSI['W']}")
        print(f"{ANSI['G']}🎯 NEXT STEPS:{ANSI['W']}")
        print(f"{ANSI['B']}{'='*80}{ANSI['W']}")
        print("1. Review the optimization results above")
        print("2. Train the final model with optimized hyperparameters:")
        print(f"   python main.py --mode train-optimized --hyperparams models/hyperopt/{args.study_name}_best_params.json")
        print("3. Or use the train_optimized script directly:")
        print(f"   python -m src.train_optimized --hyperparams models/hyperopt/{args.study_name}_best_params.json")
        
    elif mode == "sweep":
        logger.info("Starting W&B Sweep...")
        
        sweep_id = initialize_sweep(
            project=args.sweep_project,
            sweep_name=args.sweep_name,
            config_path=args.config,
            entity=args.entity
        )
        
        print(f"\n{ANSI['B']}{'='*80}{ANSI['W']}")
        print(f"{ANSI['G']}🚀 W&B SWEEP INITIALIZED{ANSI['W']}")
        print(f"{ANSI['B']}{'='*80}{ANSI['W']}")
        print(f"{ANSI['B']}Sweep ID:{ANSI['W']} {sweep_id}")
        print(f"{ANSI['B']}Next steps:{ANSI['W']}")
        print("1. Visit your W&B dashboard to monitor the sweep")
        print("2. Run an agent with:")
        print(f"   python main.py --mode sweep-agent --sweep-id {sweep_id}")
        print("3. Or run multiple agents in parallel:")
        print(f"   python main.py --mode sweep-agent --sweep-id {sweep_id} --count 5")
        print(f"4. Sweep URL: https://wandb.ai/{args.entity or 'your-username'}/{args.sweep_project}/sweeps/{sweep_id}")
        
    elif mode == "sweep-agent":
        if not args.sweep_id:
            logger.error("--sweep-id argument required for sweep-agent mode")
            logger.info("Initialize a sweep first:")
            logger.info(f"  python main.py --mode sweep --sweep-name {args.sweep_name}")
            return
        
        logger.info("Starting W&B Sweep Agent...")
        
        run_sweep_agent(
            sweep_id=args.sweep_id,
            project=args.sweep_project,
            config_path=args.config,
            max_epochs=args.max_epochs,
            count=args.count,
            entity=args.entity
        )
        logger.info("W&B Sweep Agent completed.")

def create_optimize_parser():
    """
    Create optimization-specific argument parser.
    
    Returns:
        ArgumentParser for optimization commands
    """
    parser = argparse.ArgumentParser(description="CSI-Predictor Optimization")
    
    # General arguments
    parser.add_argument("--config", help="Path to config.ini file (if not provided, will use INI_DIR from .env)")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    
    # Hyperparameter optimization specific arguments (Optuna)
    parser.add_argument("--study-name", default="csi_optimization", help="Name of the Optuna study")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for hyperparameter optimization")
    parser.add_argument("--max-epochs", type=int, default=30, help="Maximum epochs per trial during optimization")
    parser.add_argument("--sampler", default="tpe", choices=['tpe', 'random', 'cmaes'], help="Optuna sampler algorithm")
    parser.add_argument("--pruner", default="median", choices=['median', 'successive_halving', 'none'], help="Optuna pruner algorithm")
    parser.add_argument("--wandb-project", default="csi-hyperopt", help="WandB project name for hyperopt logging")
    
    # W&B Sweeps specific arguments
    parser.add_argument("--sweep-project", default="csi-sweeps", help="W&B project name for sweeps")
    parser.add_argument("--sweep-name", default="csi_sweep", help="Name for the W&B sweep")
    parser.add_argument("--sweep-id", help="W&B sweep ID (for sweep-agent mode)")
    parser.add_argument("--entity", help="W&B entity (username/team)")
    parser.add_argument("--count", type=int, help="Number of runs for sweep agent")
    
    return parser

if __name__ == "__main__":
    parser = create_optimize_parser()
    args = parser.parse_args()
    optimize_cli(args)

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 