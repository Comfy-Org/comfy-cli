import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from packaging import version

from comfy_cli._safe_exec import resolve_binary

logger = logging.getLogger(__name__)

UPDATE_CHECK_TTL_SECONDS = 24 * 60 * 60
UPDATE_CHECK_DISABLE_ENV = "COMFY_NO_UPDATE_CHECK"


def check_for_newer_pypi_version(package_name, current_version):
    """Return ``(has_newer, latest_version)`` for the named package.

    Used by the welcome banner to indicate "an upgrade is available";
    the standalone ``check_for_updates()`` side-effect that fired a
    bright blue "🔔 Update Available!" panel mid-command has been
    retired (Task 5 of the CLI UX consistency pass) — chrome should be
    one panel per command, not random unsolicited banners.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        latest_version = response.json()["info"]["version"]

        if version.parse(latest_version) > version.parse(current_version):
            return True, latest_version

        return False, current_version
    except requests.RequestException as e:
        logger.warning(f"Failed to check for updates: {e}")
        return False, current_version


class UpgradeUnavailable(RuntimeError):
    """No installer can upgrade the environment comfy-cli is running from."""

    def __init__(self, message: str, hint: str):
        super().__init__(message)
        self.hint = hint


def upgrade_cli():
    """Upgrade the ``comfy-cli`` package itself in the environment it runs from.

    A ``uv tool install`` env ships without pip, so it is upgraded through
    ``uv tool upgrade``; anything else goes through pip against the running
    interpreter. Raises ``UpgradeUnavailable`` when neither installer is usable
    and ``CalledProcessError`` if the installer fails — fail fast."""
    if (Path(sys.prefix) / "uv-receipt.toml").is_file():
        uv = resolve_binary("uv")
        if uv is None:
            raise UpgradeUnavailable(
                "comfy-cli was installed with `uv tool`, but `uv` is not on PATH.",
                hint="run `uv tool upgrade comfy-cli` from a shell where uv is available",
            )
        subprocess.run([uv, "tool", "upgrade", "comfy-cli"], check=True)
        return
    if importlib.util.find_spec("pip") is None:
        raise UpgradeUnavailable(
            f"pip is not installed in the comfy-cli environment ({sys.prefix}).",
            hint="upgrade comfy-cli with the tool that installed it (e.g. `pipx upgrade comfy-cli`)",
        )
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "comfy-cli"], check=True)


def _read_fresh_cache(cache_file, now):
    """Return the cached latest version if the cache exists and is fresh, else None."""
    if not cache_file.exists():
        return None
    payload = json.loads(cache_file.read_text())
    if now - payload["checked_at"] > UPDATE_CHECK_TTL_SECONDS:
        return None
    return payload["latest"]


def _refresh_cache(cache_file, current_version, now):
    """Query PyPI, persist the result, and return the latest version string."""
    _, latest = check_for_newer_pypi_version("comfy-cli", current_version)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"checked_at": now, "latest": latest}))
    return latest


def latest_upgrade_version(current_version, config_path):
    """Return the newer PyPI version string, or ``None`` if already current.

    Cached for ``UPDATE_CHECK_TTL_SECONDS`` under ``config_path`` so PyPI is
    hit at most once a day. Opt out entirely via ``COMFY_NO_UPDATE_CHECK``.
    Best-effort: any failure resolves to "no upgrade to show" rather than
    breaking the calling command.
    """
    if os.getenv(UPDATE_CHECK_DISABLE_ENV):
        return None

    now = time.time()
    cache_file = Path(config_path) / "update-check.json"
    try:
        latest = _read_fresh_cache(cache_file, now) or _refresh_cache(cache_file, current_version, now)
        if latest is None:
            return None
        if version.parse(latest) > version.parse(current_version):
            return latest
        return None
    except (OSError, ValueError, TypeError, KeyError) as e:
        logger.debug(f"Update check skipped: {e}")
        return None
