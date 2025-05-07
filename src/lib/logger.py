"""Centralized logging configuration for the Doctor project."""

import logging
import os
from typing import Optional


def configure_logging(
    name: str = None,
    level: str = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with consistent formatting.

    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Falls back to DOCTOR_LOG_LEVEL env var, defaults to INFO
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    # Get log level from params, env var, or default to INFO
    if level is None:
        level = os.getenv("DOCTOR_LOG_LEVEL", "INFO")

    level = getattr(logging, level.upper())

    # Configure root logger if not already configured
    if not logging.getLogger().handlers:
        # Basic configuration with consistent format
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            logging.getLogger().addHandler(file_handler)

    # Get logger for the specific module
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Configured logger instance
    """
    return configure_logging(name)
