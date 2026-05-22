import logging
from importlib.metadata import metadata

import requests
from packaging import version

logger = logging.getLogger(__name__)


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


def get_version_from_pyproject():
    package_metadata = metadata("comfy-cli")
    return package_metadata["Version"]
