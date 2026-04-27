import logging
import sys
from typing import Optional

# Default log format: includes timestamp, level, module, and message
DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# SNN-specific log levels (optional extensions beyond standard levels)
# You can use these for custom filtering, e.g., log only SNN metrics
SNN_DEBUG = 5  # Below DEBUG, for very detailed SNN internals (e.g., per-neuron states)
SNN_INFO = logging.INFO  # Standard INFO
logging.addLevelName(SNN_DEBUG, "SNN_DEBUG")


def configure_logging(level=logging.INFO):
    """
    Configure logging with default format (includes timestamp).
    
    Args:
        level: Logging level (default: logging.INFO)
    
    Returns:
        Root logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter with default format
    formatter = logging.Formatter(DEFAULT_FORMAT)
    
    # Add console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name):
    """
    Get a logger instance for the given module name.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Example usage
if __name__ == "__main__":
    # Setup logging (console only)
    configure_logging(level=logging.DEBUG)
    
    # Get a logger
    logger = get_logger(__name__)
    
    # Log messages with timestamp
    logger.info("SNN logging initialized")
    logger.debug("Debug message example")
    logger.warning("Warning message example")
    logger.error("Error message example")