"""
Logging setup for CSI-Predictor.

This module contains logging functionality extracted from the original src/utils.py file.
"""

import sys
from pathlib import Path
from loguru import logger

def setup_logging(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    """
    Setup Loguru logging with rotating file handler.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
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
    
    logger.info(f"Logging setup complete. Log files will be stored in: {log_dir}")

# Setup logging on import
setup_logging()

__version__ = "1.0.0"
__author__ = "CSI-Predictor Team" 