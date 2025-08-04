"""
Logging setup for CSI-Predictor.

This module contains logging functionality extracted from the original src/utils.py file.
"""

import sys
from pathlib import Path
from loguru import logger

# Flag to track if logging has been set up
_logging_setup = False

def setup_logging(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    """
    Setup Loguru logging with rotating file handler.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    global _logging_setup
    
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler and any existing handlers
    logger.remove()
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # Add rotating file handler
    logger.add(
        Path(log_dir) / "csi_predictor_{time:YYYY-MM-DD}.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )
    
    _logging_setup = True
    logger.info(f"Logging setup complete. Log files will be stored in: {log_dir}")

# Don't setup logging automatically - let the config system handle it
# This prevents creating a logs folder in the project root

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 