import logging
import sys

DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def configure_logging(level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter(DEFAULT_FORMAT)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
