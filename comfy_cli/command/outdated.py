"""``comfy outdated`` — read-only installed-vs-latest report.

Reports the installed vs. latest version for ComfyUI **core** and every
installed **custom node pack**, so a stale install can be spotted before it
silently degrades agents. Nothing here mutates the workspace — it only reads
git metadata / ``pyproject.toml`` and queries public APIs.

Sources of "latest":
- **Core**: the highest stable semver tag found locally (mirroring
  :func:`comfy_cli.command.install._resolve_latest_tag_from_local`), falling
  back to the GitHub ``releases/latest`` tag (reusing
  :func:`comfy_cli.command.install.get_latest_release`) only when the local
  checkout has no usable tags. The local-first order matters: GitHub's
  ``releases/latest`` is a maintainer-set mutable flag and can point at a tag
  that isn't actually the highest semver release.
- **Registry packs** (a pack whose ``pyproject.toml`` carries a registry node
  id + version): the Comfy registry API (reusing
  :class:`comfy_cli.registry.RegistryAPI`).
- **Git packs** (a bare git clone, no registry metadata): ``git ls-remote``
  HEAD compared against the local HEAD.

Latest-version lookups are cached for 1h under
``~/.cache/comfy-cli/outdated.json`` so agents can poll cheaply; ``--refresh``
bypasses the cache. Any network failure degrades to ``latest: null`` + a
warning and exit 0 — a report that can't reach the network is still useful.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import semver

from comfy_cli._safe_exec import resolve_required_binary
from comfy_cli.command.pack_scan import iter_pack_dirs as _iter_pack_dirs
from comfy_cli.command.pack_scan import read_pyproject as _read_pyproject
from comfy_cli.file_utils import atomic_write_text
from comfy_cli.registry import RegistryAPI

CACHE_TTL_SECONDS = 3600  # 1 hour
GIT_TIMEOUT_SECONDS = 10

# Matches the trailing ``-<N>-g<sha>`` that ``git describe`` appends when the
# checkout is N commits past the nearest tag (e.g. ``v0.3.40-5-gdeadbee``).
_DESCRIBE_SUFFIX_RE = re.compile(r"-\d+-g[0-9a-f]+$")

# Matches a top-level ``__version__ = "x.y.z"`` in ComfyUI's comfyui_version.py.
_VERSION_ASSIGN_RE = re.compile(
    r"""^__version__\s*=\s*(?:"([^"\n]*)"|'([^'\n]*)')""",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    """Where the latest-version cache lives. XDG-respecting."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / "outdated.json"


def _load_cache() -> dict[str, Any]:
    path = _cache_path()
    try:
        data = json.loads(path.read_bytes())
    except (OSError, ValueError):
        # ValueError covers JSONDecodeError *and* the UnicodeDecodeError that a
        # non-UTF-8 (corrupt) cache file raises — either way, start fresh.
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache(cache: dict[str, Any]) -> None:
    path = _cache_path()
    try:
        # Write-then-rename (tier 4, regenerable cache — see the write policy in
        # comfy_cli/file_utils.py): an interrupt mid-write must not leave a
        # truncated file that the next `_load_cache` silently resets to `{}`.
        # The helper creates the parent dir, writes UTF-8 (`_load_cache` decodes
        # bytes as JSON, which only accepts UTF-8/16/32), and cleans up its own
        # temp file on failure.
        atomic_write_text(path, json.dumps(cache), fsync=False)
    except OSError:
        # A read-only cache dir must never break a read-only report.
        pass


def _cache_get(cache: dict[str, Any], key: str) -> Any | None:
    """Return a fresh (< TTL) cached value for *key*, else ``None``."""
    entry = cache.get(key)
    if not isinstance(entry, dict):
        return None
    ts = entry.get("ts")
    if not isinstance(ts, int | float) or (time.time() - ts) > CACHE_TTL_SECONDS:
        return None
    return entry.get("value")


def _cache_set(cache: dict[str, Any], key: str, value: Any) -> None:
    cache[key] = {"value": value, "ts": time.time()}


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _normalize_version(v: str | None) -> str | None:
    if not v:
        return v
    v = _DESCRIBE_SUFFIX_RE.sub("", v.strip())
    return v.lstrip("vV")


def _is_outdated(installed: str | None, latest: str | None) -> bool:
    """True when *installed* is provably older than *latest*.

    Unknowns (either side missing) are never flagged outdated. Comparison:

    - both parse as PEP 440 versions → semantic ``<`` (a checkout *ahead* of the
      latest tag, ``v0.3.41-5-g…`` vs ``v0.3.41``, normalizes equal → not flagged);
    - neither parses (e.g. two git SHAs, for a git pack) → any difference means
      the local HEAD is behind the remote HEAD → outdated;
    - exactly one parses (e.g. a shallow/no-tag core checkout whose "installed"
      is a bare SHA vs a version tag) → genuinely incomparable, so *not* flagged
      rather than guessing a false positive.
    """
    if not installed or not latest:
        return False
    ni, nl = _normalize_version(installed), _normalize_version(latest)
    if ni == nl:
        return False

    from packaging.version import InvalidVersion, parse

    pi = pl = None
    try:
        pi = parse(ni)
    except InvalidVersion:
        pass
    try:
        pl = parse(nl)
    except InvalidVersion:
        pass

    if pi is not None and pl is not None:
        return pi < pl
    if pi is None and pl is None:
        return ni != nl
    return False


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


# Git runs with ``cwd`` inside a custom-node pack we do not control, so it honors
# that pack's ``.git/config`` and ``origin`` URL. A pack shipped as an archive
# with an embedded ``.git`` could otherwise turn a read-only version check into
# arbitrary code execution. Hardening, applied to *every* git call, neutralizes
# each knob a malicious repo config could use to run a program during a plain
# read, while keeping the legitimate https/ssh transports working:
#   * ``GIT_ALLOW_PROTOCOL`` drops ``ext::`` (runs a shell) and ``git://`` (whose
#     ``core.gitProxy`` runs a program), leaving file/http/https/ssh.
#   * command-line ``-c`` overrides (which beat repo-local config) clear the
#     ``credential.helper`` and ``core.fsmonitor`` program hooks.
#   * ``GIT_SSH_COMMAND=ssh`` (env beats ``core.sshCommand``) pins ssh to the
#     default client, so a pack can't hijack it — yet ssh remotes still resolve.
#   * ``GIT_TERMINAL_PROMPT=0`` keeps a credential-needing remote from blocking.
_GIT_HARDENING: list[str] = [
    "-c",
    "credential.helper=",
    "-c",
    "core.fsmonitor=",
    "-c",
    "protocol.ext.allow=never",
    "-c",
    "protocol.git.allow=never",
]
_GIT_SAFE_ENV = {
    "GIT_ALLOW_PROTOCOL": "file:http:https:ssh",
    "GIT_SSH_COMMAND": "ssh",
    "GIT_TERMINAL_PROMPT": "0",
}


def _git_output(args: list[str], cwd: str) -> str | None:
    """Run ``git <args>`` in *cwd*, returning stripped stdout or ``None``.

    Hardened against a malicious pack ``.git/config``/``origin`` — see
    ``_GIT_HARDENING`` / ``_GIT_SAFE_ENV``.
    """
    try:
        # ``resolve_required_binary`` raises ``BinaryNotFoundError`` (a
        # ``FileNotFoundError``), which the ``OSError`` arm below already turns
        # into the same ``None`` a missing ``git`` produced before.
        out = subprocess.run(
            [resolve_required_binary("git"), *_GIT_HARDENING, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=True,
            env={**os.environ, **_GIT_SAFE_ENV},
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return out.stdout.strip() or None


def _is_git_checkout(path: str) -> bool:
    return _git_output(["rev-parse", "--is-inside-work-tree"], path) == "true"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _core_installed(comfy_path: str) -> tuple[str | None, str | None]:
    """Return ``(installed_version, commit)`` for the core install.

    Prefers ``git describe --tags`` (+ short HEAD) for a git checkout, falling
    back to ``comfyui_version.py``'s ``__version__`` then ``pyproject.toml``.
    """
    if _is_git_checkout(comfy_path):
        described = _git_output(["describe", "--tags", "--always"], comfy_path)
        commit = _git_output(["rev-parse", "--short", "HEAD"], comfy_path)
        if described:
            return described, commit

    # Fall back to the packaged version marker.
    version_file = Path(comfy_path) / "comfyui_version.py"
    try:
        m = _VERSION_ASSIGN_RE.search(version_file.read_text(encoding="utf-8"))
        if m:
            return (m.group(1) or m.group(2)), None
    except OSError:
        pass

    cfg = _read_pyproject(os.path.join(comfy_path, "pyproject.toml"))
    if cfg is not None and cfg.project.name:
        return cfg.project.version, None
    return None, None


def _core_latest_from_local_tags(comfy_path: str) -> str | None:
    """Highest stable semver git tag already known to the core checkout.

    Mirrors :func:`comfy_cli.command.install._resolve_latest_tag_from_local`:
    ``git fetch --tags`` (best-effort — a stale/offline tag list is still
    useful) then pick the highest non-prerelease semver tag. Uses this
    module's hardened :func:`_git_output` rather than install.py's own
    subprocess calls, since ``comfy_path`` is already run through it by
    :func:`_core_installed`.

    Returns ``None`` when *comfy_path* isn't a git checkout or has no stable
    semver tags, so the caller can fall back to the GitHub API.
    """
    if not _is_git_checkout(comfy_path):
        return None
    _git_output(["fetch", "--tags", "--quiet"], comfy_path)
    tags = _git_output(["tag", "--list"], comfy_path)
    if not tags:
        return None

    best: tuple[semver.VersionInfo, str] | None = None
    for line in tags.splitlines():
        tag = line.strip()
        if not tag:
            continue
        try:
            parsed = semver.VersionInfo.parse(tag.lstrip("vV"))
        except ValueError:
            continue
        if parsed.prerelease:
            continue
        if best is None or parsed > best[0]:
            best = (parsed, tag)
    return best[1] if best else None


def _core_latest(comfy_path: str, cache: dict[str, Any], refresh: bool, warn: Callable[[str], None]) -> str | None:
    if not refresh:
        cached = _cache_get(cache, "core")
        if cached is not None:
            return cached

    # Prefer the highest local semver tag over GitHub's `releases/latest` —
    # see `_core_latest_from_local_tags` / module docstring for why.
    local_tag = _core_latest_from_local_tags(comfy_path)
    if local_tag is not None:
        _cache_set(cache, "core", local_tag)
        return local_tag

    # No usable local tags (e.g. not a git checkout, or a shallow/tagless
    # clone) — fall back to install.py's rate-limit-aware fetcher
    # (GITHUB_TOKEN, forks, 403/429).
    from comfy_cli.command.install import get_latest_release

    try:
        # get_latest_release prints its own error line to stdout via rich on a
        # network failure; redirect that to stderr so it can't corrupt the
        # single-line JSON envelope stdout contract (mirrors _read_pyproject).
        with contextlib.redirect_stdout(sys.stderr):
            release = get_latest_release("comfyanonymous", "ComfyUI")
    except Exception as e:  # noqa: BLE001 - never let a network hiccup abort the report
        warn(f"could not fetch latest ComfyUI release: {e}")
        return None
    if release is None:
        warn("could not fetch latest ComfyUI release (network or rate limit)")
        return None
    # get_latest_release returns a GithubRelease TypedDict (a plain dict).
    tag = release.get("tag")
    if not tag:
        warn("latest ComfyUI release had no tag")
        return None
    _cache_set(cache, "core", tag)
    return tag


# ---------------------------------------------------------------------------
# Custom node packs
# ---------------------------------------------------------------------------


def _registry_latest(
    node_id: str,
    cache: dict[str, Any],
    refresh: bool,
    registry_api: RegistryAPI,
    warn: Callable[[str], None],
) -> str | None:
    key = f"pack:{node_id}"
    if not refresh:
        cached = _cache_get(cache, key)
        if cached is not None:
            return cached
    try:
        # get_node, not install_node: the install endpoint records an
        # installation + analytics event server-side on every call, which a
        # read-only report must not inflate.
        node = registry_api.get_node(node_id)
    except Exception as e:  # noqa: BLE001 - registry unreachable → unknown, not fatal
        warn(f"could not fetch latest version for pack '{node_id}': {e}")
        return None
    latest = getattr(getattr(node, "latest_version", None), "version", None)
    if latest:
        _cache_set(cache, key, latest)
    return latest


def _git_pack_info(
    pack_dir: Path,
    cache: dict[str, Any],
    refresh: bool,
    warn: Callable[[str], None],
) -> dict[str, Any]:
    """Compare a git pack's local HEAD against its remote's HEAD."""
    name = pack_dir.name
    installed = _git_output(["rev-parse", "HEAD"], str(pack_dir))
    latest = None
    key = f"pack:git:{pack_dir}"
    cached = _cache_get(cache, key) if not refresh else None
    if cached is not None:
        latest = cached
    else:
        remote = _git_output(["remote", "get-url", "origin"], str(pack_dir))
        if remote:
            # ``--`` stops a ``origin`` URL starting with ``-`` from being
            # parsed as a git option (option-injection); transports are
            # further restricted in ``_git_output``.
            ls = _git_output(["ls-remote", "--", remote, "HEAD"], str(pack_dir))
            if ls:
                latest = ls.split()[0]
                _cache_set(cache, key, latest)
            else:
                warn(f"could not reach git remote for pack '{name}'")
        else:
            warn(f"no origin remote for git pack '{name}'")
    return {
        "name": name,
        "source": "git",
        "installed": installed,
        "latest": latest,
        "outdated": _is_outdated(installed, latest),
    }


def _pack_info(
    pack_dir: Path,
    cache: dict[str, Any],
    refresh: bool,
    registry_api: RegistryAPI,
    warn: Callable[[str], None],
) -> dict[str, Any]:
    name = pack_dir.name
    pyproject = pack_dir / "pyproject.toml"
    is_git = _is_git_checkout(str(pack_dir))

    # Registry pack: pyproject carries a node id + version → registry API.
    if pyproject.is_file():
        cfg = _read_pyproject(str(pyproject))
        if cfg is not None and cfg.project.name:
            node_id = cfg.project.name
            installed = cfg.project.version
            latest = _registry_latest(node_id, cache, refresh, registry_api, warn)
            # Many git-installed packs also ship a pyproject but aren't in the
            # registry (latest unknown). Rather than report them as permanently
            # "unknown", fall back to the git HEAD comparison when we can.
            if latest is None and is_git:
                return _git_pack_info(pack_dir, cache, refresh, warn)
            return {
                "name": node_id,
                "source": "registry",
                "installed": installed,
                "latest": latest,
                "outdated": _is_outdated(installed, latest),
            }

    # Git pack: compare local HEAD against the remote's HEAD.
    if is_git:
        return _git_pack_info(pack_dir, cache, refresh, warn)

    # Neither: report what we can, latest unknown.
    return {
        "name": name,
        "source": "unknown",
        "installed": None,
        "latest": None,
        "outdated": False,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    comfy_path: str | None,
    *,
    refresh: bool = False,
    registry_api: RegistryAPI | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build the outdated report. Returns ``(report, warnings)``.

    Pure enough to unit-test: inject ``registry_api`` (needs ``get_node``)
    and ``now``; all network paths degrade to ``latest: null`` + a warning.
    """
    warnings: list[str] = []
    warn = warnings.append
    registry_api = registry_api or RegistryAPI()
    cache = _load_cache()

    if comfy_path and os.path.isdir(comfy_path):
        core_installed, core_commit = _core_installed(comfy_path)
        core_latest = _core_latest(comfy_path, cache, refresh, warn)
        packs = [
            _pack_info(p, cache, refresh, registry_api, warn)
            for p in _iter_pack_dirs(Path(comfy_path) / "custom_nodes")
        ]
    else:
        warn(f"ComfyUI workspace not found at {comfy_path!r}")
        core_installed, core_commit, core_latest, packs = None, None, None, []

    _save_cache(cache)

    checked_at = (now or datetime.now(timezone.utc)).isoformat()
    report = {
        "core": {
            "installed": core_installed,
            "commit": core_commit,
            "latest": core_latest,
            "outdated": _is_outdated(core_installed, core_latest),
        },
        "packs": packs,
        "checked_at": checked_at,
    }
    return report, warnings


def _render_pretty(renderer, report: dict[str, Any]) -> None:
    from rich.markup import escape
    from rich.table import Table

    def _fmt(installed: Any, latest: Any, outdated: bool) -> tuple[str, str]:
        # Pack names/versions come from the filesystem and pyproject; escape any
        # ``[...]`` so a value like ``foo[/]`` can't raise a rich MarkupError and
        # crash the pretty report.
        i = escape(str(installed)) if installed is not None else "[dim]?[/dim]"
        latest_str = escape(str(latest)) if latest is not None else "[dim]unknown[/dim]"
        if outdated:
            return f"[yellow]{i}[/yellow]", f"[bold green]{latest_str}[/bold green]"
        return i, latest_str

    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("component")
    tbl.add_column("installed")
    tbl.add_column("latest")
    tbl.add_column("status")

    core = report["core"]
    ci, cl = _fmt(core["installed"], core["latest"], core["outdated"])
    tbl.add_row("[bold]ComfyUI (core)[/bold]", ci, cl, _status(core["outdated"], core["latest"]))

    for pack in report["packs"]:
        pi, pl = _fmt(pack["installed"], pack["latest"], pack["outdated"])
        tbl.add_row(escape(pack["name"]), pi, pl, _status(pack["outdated"], pack["latest"]))

    renderer.console().print(tbl)
    n_outdated = int(core["outdated"]) + sum(1 for p in report["packs"] if p["outdated"])
    if n_outdated:
        renderer.print(f"[yellow]{n_outdated} component(s) outdated.[/yellow]")
    else:
        renderer.print("[green]Everything is up to date.[/green]")


def _status(outdated: bool, latest: Any) -> str:
    if outdated:
        return "[bold yellow]outdated[/bold yellow]"
    if latest is None:
        return "[dim]unknown[/dim]"
    return "[green]up to date[/green]"


def execute(renderer, comfy_path: str | None, *, refresh: bool = False) -> None:
    """Entry point wired from ``comfy outdated`` in cmdline.py."""
    report, warnings = build_report(comfy_path, refresh=refresh)
    if renderer.is_pretty():
        _render_pretty(renderer, report)
    for w in warnings:
        # Warnings embed pack names/error text, but ``renderer.warn`` escapes
        # markup itself now — escaping again here would render the backslashes.
        renderer.warn(w)
    renderer.emit(report, command="outdated")
