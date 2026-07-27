"""Logging configuration for the application"""
import logging
import os
from datetime import datetime
from .config import Config

def setup_logger(name: str = None) -> logging.Logger:
    """Set up and return a configured logger instance"""
    
    # Create logger
    logger = logging.getLogger(name or __name__)
    
    # Skip if handlers already configured
    if logger.handlers:
        return logger
        
    level = getattr(logging, Config.LOG_LEVEL.upper(), None)
    if not isinstance(level, int):
        logging.getLogger(name or __name__).warning(
            f"Invalid LOG_LEVEL '{Config.LOG_LEVEL}', falling back to INFO"
        )
        level = logging.INFO
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create handlers
    # File handler
    log_file = os.path.join(
        Config.LOGS_DIR,
        f'app_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Create default logger
logger = setup_logger('stock_analysis')