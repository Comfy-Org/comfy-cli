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
CLI_UPGRADE_TIMEOUT_SECONDS = 600


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
    """The running installation cannot be upgraded automatically."""

    def __init__(self, message: str, hint: str):
        super().__init__(message)
        self.hint = hint


def _check_uv_tool_pin(receipt: Path) -> None:
    """Preserve exact primary-tool constraints rather than reporting a no-op upgrade."""
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    from tomlkit import parse
    from tomlkit.exceptions import ParseError

    tool_root = receipt.parent.parent
    hint = (
        "Review the receipt. To change or remove a pin, explicitly reinstall using your chosen requirement "
        "(for example `uv tool install --force comfy-cli` to remove it); "
        f"first set UV_TOOL_DIR to {tool_root} to target this installation."
    )
    pinned = False
    try:
        tool = parse(receipt.read_text(encoding="utf-8")).unwrap().get("tool")
        if not isinstance(tool, dict):
            raise ValueError("missing tool table")
        for field in ("requirements", "constraints"):
            rows = tool.get(field, [])
            if not isinstance(rows, list):
                raise ValueError("invalid requirement list")
            for row in rows:
                if isinstance(row, str):
                    requirement = Requirement(row)
                    name, specifiers = requirement.name, requirement.specifier
                elif isinstance(row, dict):
                    name = row.get("name")
                    if not isinstance(name, str):
                        raise ValueError("missing requirement name")
                    parsed_name = Requirement(name)
                    if parsed_name.extras or parsed_name.url or parsed_name.specifier or parsed_name.marker:
                        raise ValueError("invalid package name")
                    if canonicalize_name(name) != "comfy-cli":
                        continue
                    specifier = row.get("specifier", "")
                    if not isinstance(specifier, str):
                        raise ValueError("invalid primary specifier")
                    specifiers = SpecifierSet(specifier)
                else:
                    raise ValueError("invalid requirement row")
                if canonicalize_name(name) == "comfy-cli" and any(
                    spec.operator == "===" or (spec.operator == "==" and not spec.version.endswith(".*"))
                    for spec in specifiers
                ):
                    pinned = True
    except (OSError, ValueError, ParseError) as error:
        raise UpgradeUnavailable("The running uv tool receipt could not be read or understood.", hint=hint) from error
    if pinned:
        raise UpgradeUnavailable(
            "The running comfy-cli installation has an exact uv version pin; this command will not remove it.",
            hint=hint,
        )


def upgrade_cli():
    """Upgrade the ``comfy-cli`` package itself in the environment it runs from.

    A ``uv tool install`` env ships without pip, so it is upgraded through
    ``uv tool upgrade``; anything else goes through pip against the running
    interpreter. Exact uv pins remain unchanged. Raises ``UpgradeUnavailable``
    for unusable installers or receipts, and propagates installer exit, start,
    or timeout failures to the command."""
    from comfy_cli.output import get_renderer

    installer_stdout = sys.stderr if get_renderer().is_json() else None
    receipt = Path(sys.prefix) / "uv-receipt.toml"
    if receipt.is_file():
        uv = resolve_binary("uv")
        if uv is None:
            raise UpgradeUnavailable(
                "comfy-cli was installed with `uv tool`, but `uv` is not on PATH.",
                hint="run `uv tool upgrade comfy-cli` from a shell where uv is available",
            )
        _check_uv_tool_pin(receipt)
        # Upgrade the running installation even if its original tool-dir override
        # is absent now, or the current shell selects another tools directory.
        installer_env = os.environ.copy()
        installer_env["UV_TOOL_DIR"] = str(Path(sys.prefix).parent)
        subprocess.run(
            [uv, "tool", "upgrade", "comfy-cli"],
            check=True,
            timeout=CLI_UPGRADE_TIMEOUT_SECONDS,
            env=installer_env,
            stdout=installer_stdout,
        )
        return
    if importlib.util.find_spec("pip") is None:
        raise UpgradeUnavailable(
            f"pip is not installed in the comfy-cli environment ({sys.prefix}).",
            hint="upgrade comfy-cli with the tool that installed it (e.g. `pipx upgrade comfy-cli`)",
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "comfy-cli"],
        check=True,
        timeout=CLI_UPGRADE_TIMEOUT_SECONDS,
        stdout=installer_stdout,
    )


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
