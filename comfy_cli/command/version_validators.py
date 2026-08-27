"""Typer ``callback=`` validators for ``--version`` flags.

Kept in a module of their own because ``cmdline.py`` needs them at
decoration time (typer reads a callback's signature when the command is
defined), and their previous home, ``command/install.py``, imports
``requests`` and the custom-nodes machinery, none of which a
``comfy --version`` call should pay for.
"""

from __future__ import annotations

import semver
import typer


def validate_version(version: str) -> str | None:
    """
    Validates the version string as 'latest', 'nightly', or a semantically version number.

    Args:
    version (str): The version string to validate.

    Returns:
    Optional[str]: The validated version string, or None if invalid.

    Raises:
    ValueError: If the version string is invalid.
    """
    if version.lower() in ["nightly", "latest"]:
        return version.lower()

    # Remove 'v' prefix if present
    if version.startswith("v"):
        version = version[1:]

    try:
        semver.VersionInfo.parse(version)
        return version
    except ValueError as exc:
        raise ValueError(
            f"Invalid version format: {version}. "
            "Please use 'nightly', 'latest', or a valid semantic version (e.g., '1.2.3')."
        ) from exc


def validate_optional_version(version: str | None) -> str | None:
    """Typer callback for an *optional* ``--version`` flag.

    ``validate_version`` is written for a flag that always has a value (``comfy
    install --version`` defaults to ``nightly``). ``comfy update`` treats the
    flag as opt-in, so ``None`` must pass through untouched. Invalid input is
    re-raised as ``typer.BadParameter`` so a headless caller gets the standard
    CLI usage error instead of a traceback.
    """
    if version is None:
        return None
    try:
        return validate_version(version)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
