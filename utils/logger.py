"""
Logging configuration for the application
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "30 days"
):
    """
    Setup application logger with file and console output.
    
    Args:
        log_file: Path to log file
        level: Logging level
        rotation: Log rotation size
        retention: Log retention period
    """
    # Remove default logger
    logger.remove()
    
    # Console logger
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logger (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    return logger


# Create default logger instance
default_logger = setup_logger()
