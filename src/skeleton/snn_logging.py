import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# Default log format: includes timestamp, level, module, and message
DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# SNN-specific log levels (optional extensions beyond standard levels)
# You can use these for custom filtering, e.g., log only SNN metrics
SNN_DEBUG = 5  # Below DEBUG, for very detailed SNN internals (e.g., per-neuron states)
SNN_INFO = logging.INFO  # Standard INFO
logging.addLevelName(SNN_DEBUG, "SNN_DEBUG")

class SNNLogger:
    """
    Centralized logger for the SNN project.
    Handles setup, formatting, and SNN-specific logging helpers.
    """

    _instance: Optional[logging.Logger] = None
    _configured = False

    @classmethod
    def setup_logging(
        cls,
        level: str = "INFO",
        log_file: Optional[str] = None,
        console: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB per log file
        backup_count: int = 5,
        format_str: str = DEFAULT_FORMAT
    ) -> logging.Logger:
        """
        Configure logging once for the entire application.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL, or SNN_DEBUG).
            log_file: Path to log file (optional). If provided, rotates on size.
            console: Enable console output.
            max_bytes: Max size per log file before rotation.
            backup_count: Number of backup log files to keep.
            format_str: Custom log format string.

        Returns:
            Root logger instance.
        """
        if cls._configured:
            return cls._instance

        # Map string levels to logging constants
        level_map = {
            "SNN_DEBUG": SNN_DEBUG,
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        log_level = level_map.get(level.upper(), logging.INFO)

        # Create formatter
        formatter = logging.Formatter(format_str)

        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler with rotation if log_file is provided
        if log_file:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._instance = logger
        cls._configured = True
        return logger

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a specific module/component.

        Args:
            name: Name of the logger (e.g., __name__ for the calling module).

        Returns:
            Logger instance.
        """
        if not cls._configured:
            cls.setup_logging()  # Default setup if not configured
        return logging.getLogger(name)

    @staticmethod
    def log_snn_metrics(logger: logging.Logger, metrics: dict):
        """
        Helper to log SNN-specific metrics (e.g., spike rates, membrane stats).

        Args:
            logger: Logger instance.
            metrics: Dict of metrics, e.g., {"spike_rate": 0.15, "avg_membrane": 0.8}.
        """
        if logger.isEnabledFor(logging.INFO):
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"SNN Metrics - {metrics_str}")

    @staticmethod
    def log_pipeline_stage(logger: logging.Logger, stage: str, duration: Optional[float] = None, **kwargs):
        """
        Helper to log pipeline stages (e.g., "Event Buffer", "GPU Kernel Launch").

        Args:
            logger: Logger instance.
            stage: Stage name.
            duration: Optional duration in seconds.
            **kwargs: Additional key-value pairs to log.
        """
        msg = f"Pipeline Stage: {stage}"
        if duration is not None:
            msg += f" - Duration: {duration:.4f}s"
        if kwargs:
            extras = ", ".join(f"{k}: {v}" for k, v in kwargs.items())
            msg += f" - {extras}"
        logger.info(msg)

    @staticmethod
    def log_error_with_context(logger: logging.Logger, error: Exception, context: str = ""):
        """
        Helper to log errors with context (e.g., during GPU operations).

        Args:
            logger: Logger instance.
            error: Exception object.
            context: Additional context string.
        """
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)

# Convenience functions for easy import
def setup_logging(**kwargs) -> logging.Logger:
    """Shortcut to SNNLogger.setup_logging()."""
    return SNNLogger.setup_logging(**kwargs)

def get_logger(name: str) -> logging.Logger:
    """Shortcut to SNNLogger.get_logger()."""
    return SNNLogger.get_logger(name)

# Example usage (can be removed in production)
if __name__ == "__main__":
    # Setup logging to file and console
    setup_logging(level="DEBUG", log_file="logs/snn_pipeline.log")

    # Get a logger for this module
    logger = get_logger(__name__)

    # Log some examples
    logger.info("SNN logging initialized")
    log_pipeline_stage(logger, "Event Camera", duration=0.05, events_processed=1000)
    log_snn_metrics(logger, {"spike_rate": 0.12, "energy_consumption": 45.6})
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_error_with_context(logger, e, "Test context")