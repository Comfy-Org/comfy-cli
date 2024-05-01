import logging
import os

"""
This module provides logging utilities for the CLI.

Note: we could potentially change the logging library or the way we log messages in the future.
Therefore, it's a good idea to encapsulate logging-related code in a separate module.
"""


def setup_logging():
    # TODO: consider supporting different ways of outputting logs
    # Note: by default, the log level is set to INFO
    log_level = os.getenv("LOG_LEVEL", "WARN").upper()
    logging.basicConfig(
        level=log_level,
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
