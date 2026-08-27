"""Deprecated spelling of :mod:`comfy_cli.builder_api`, kept for one release.

The builder's public API renamed distributions to builds, and the module
followed. Importing this name still works but warns; switch to
``comfy_cli.builder_api``.
"""

from __future__ import annotations

import warnings

from comfy_cli.builder_api import BuilderAuthError, BuilderClient

__all__ = ["BuilderAuthError", "BuilderClient"]

warnings.warn(
    "comfy_cli.distribution_api is deprecated; import comfy_cli.builder_api instead",
    DeprecationWarning,
    stacklevel=2,
)
