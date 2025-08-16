"""
Logging utilities for the environment package
"""

import logging
import sys
from typing import Optional


def setup_environment_logging(
    level: str = "INFO",
    format_string: Optional[str] = None,
    include_extra: bool = True,
) -> logging.Logger:
    """
    Set up structured logging for the environment package

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_string: Custom format string
        include_extra: Whether to include extra fields in logs

    Returns:
        Configured logger for the environment package
    """

    if format_string is None:
        if include_extra:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s" " [%(filename)s:%(lineno)d]"
            )
        else:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger for the environment package
    logger = logging.getLogger("opencontext.environment")
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that formats extra fields nicely
    """

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if extra:
            # Format extra fields as key=value pairs
            extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
            msg = f"{msg} | {extra_str}"
        return msg, kwargs


def get_environment_logger(name: Optional[str] = None) -> StructuredLoggerAdapter:
    """
    Get a logger for environment components with structured formatting

    Args:
        name: Optional specific name for the logger

    Returns:
        Configured logger
    """
    if name:
        logger_name = f"opencontext.environment.{name}"
    else:
        logger_name = "opencontext.environment"

    base_logger = logging.getLogger(logger_name)
    return StructuredLoggerAdapter(base_logger, {})


# Example usage patterns:
"""
# Basic setup
from opencontext.environment.logging_utils import setup_environment_logging
setup_environment_logging(level="DEBUG")

# In your class
from opencontext.environment.logging_utils import get_environment_logger

class MyClass:
    def __init__(self):
        self.logger = get_environment_logger("my_component")
        
    def some_method(self):
        self.logger.info(
            "Processing started",
            extra={"input_size": 100, "mode": "fast"}
        )
"""
