"""
This module provides logging utilities for the CLI.

Note: we could potentially change the logging library or the way we log messages in the future.
Therefore, it's a good idea to encapsulate logging-related code in a separate module.
"""

import logging
import os


def setup_logging():
    # TODO: consider supporting different ways of outputting logs
    # Note: by default, the log level is set to WARN
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level_key = os.getenv("LOG_LEVEL", "ERROR").upper()
    logging.basicConfig(
        level=log_levels.get(log_level_key, logging.WARNING),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def debug(message):
    logging.debug(message)


def info(message):
    logging.info(message)


def warning(message):
    logging.warning(message)


def error(message):
    logging.error(message)
    # TODO: consider tracking errors to Mixpanel as well.
