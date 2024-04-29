"""
Module for utility functions.
"""

import sys
from comfy_cli import constants


def singleton(cls):
    """
    Decorator that implements the Singleton pattern for the decorated class.

    e.g.
    @singleton
    class MyClass:
        pass

    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def get_os():
    if 'win' in sys.platform:
        return constants.OS.WINDOWS
    elif sys.platform == 'darwin':
        return constants.OS.MACOS

    return constants.OS.LINUX
