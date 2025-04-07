"""
Logging utilities for DeepBridge.
Provides a standardized logging system across the entire library.
"""

import logging
import sys
from typing import Optional

# Define default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Singleton logger instance
_logger_instance = None

class DeepBridgeLogger:
    """
    Centralized logging service for DeepBridge.
    Implements a singleton pattern to ensure consistency across the library.
    """
    
    def __init__(self, name: str = "deepbridge", level: int = logging.INFO):
        """
        Initialize the logger with name and level.
        
        Args:
            name: Logger name
            level: Initial logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create console handler if not already added
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(DEFAULT_FORMAT)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Track verbosity
        self.verbose = level <= logging.INFO
            
    @property
    def level(self) -> int:
        """Get current logging level"""
        return self.logger.level
        
    def set_level(self, level: int) -> None:
        """
        Set the logging level.
        
        Args:
            level: Logging level (use logging.DEBUG, logging.INFO, etc.)
        """
        self.logger.setLevel(level)
        self.verbose = level <= logging.INFO
    
    def set_verbose(self, verbose: bool) -> None:
        """
        Set verbosity based on boolean flag.
        
        Args:
            verbose: If True, sets level to INFO, otherwise to WARNING
        """
        self.verbose = verbose
        self.set_level(logging.INFO if verbose else logging.WARNING)
        
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)
        
    def exception(self, message: str) -> None:
        """Log exception with traceback"""
        self.logger.exception(message)


def get_logger(name: str = "deepbridge", level: Optional[int] = None) -> DeepBridgeLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        level: Logging level (if None, uses existing level)
        
    Returns:
        DeepBridgeLogger instance
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = DeepBridgeLogger(name, level or logging.INFO)
    elif level is not None:
        _logger_instance.set_level(level)
        
    return _logger_instance