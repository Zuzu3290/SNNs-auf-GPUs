"""
SNN Skeleton Package
====================
Provides core utilities for the SNN pipeline, including configuration,
logging, and the main pipeline execution.
"""

from .snn_config import cfg
from .snn_logging import setup_logging, get_logger, SNNLogger
from .pipeline import run_pipeline

__all__ = [
    "cfg",
    "setup_logging",
    "get_logger",
    "SNNLogger",
    "run_pipeline",
]

